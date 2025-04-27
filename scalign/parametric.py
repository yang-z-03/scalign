
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.utils import check_random_state
from pynndescent import NNDescent

from umap import UMAP
from umap.umap_ import fuzzy_simplicial_set, find_ab_params, spectral_layout, noisy_scale_coords
import os
import pickle

# see if torch is compiled with cuda.
ACCEL = 'cpu'
if torch.cuda.is_available():
    ACCEL = 'gpu'


def get_graph_elements(graph_, n_epochs):

    graph = graph_.tocoo()
    # eliminate duplicate entries by summing them together
    graph.sum_duplicates()
    # number of vertices in dataset
    n_vertices = graph.shape[1]
    # get the number of epochs based on the size of the dataset
    if n_epochs is None:
        # For smaller datasets we can use more epochs
        if graph.shape[0] <= 10000:
            n_epochs = 500
        else:
            n_epochs = 200

    # remove elements with very low probability
    graph.data[graph.data < (graph.data.max() / float(n_epochs))] = 0.0
    graph.eliminate_zeros()
    # get epochs per sample based upon edge probability
    epochs_per_sample = n_epochs * graph.data

    head = graph.row
    tail = graph.col
    weight = graph.data

    return graph, epochs_per_sample, head, tail, weight, n_vertices


class umap_dataset(Dataset):
    
    def __init__(
        self, data, graph_, n_epochs = 200, 
        landmark_data = None, landmark_embedding = None
    ):
        graph, epochs_per_sample, head, tail, weight, n_vertices = get_graph_elements(
            graph_, n_epochs
        )

        self.edges_to_exp, self.edges_from_exp = (
            np.repeat(head, epochs_per_sample.astype("int")),
            np.repeat(tail, epochs_per_sample.astype("int")),
        )
        shuffle_mask = np.random.permutation(np.arange(len(self.edges_to_exp)))
        self.edges_to_exp = self.edges_to_exp[shuffle_mask].astype(np.int64)
        self.edges_from_exp = self.edges_from_exp[shuffle_mask].astype(np.int64)
        self.data = torch.Tensor(data)

        self.use_landmark = False
        if landmark_data is not None and landmark_embedding is not None:
            self.use_landmark = True
            self.landmark_data = landmark_data
            self.landmark_embedding = landmark_embedding

    
    def __len__(self):
        return int(self.data.shape[0])

    
    def __getitem__(self, index):
        edges_to_exp = self.data[self.edges_to_exp[index]]
        edges_from_exp = self.data[self.edges_from_exp[index]]
        return (edges_to_exp, edges_from_exp)


class layout_dataset(Dataset):

    def __init__(self, data, embeddings):
        self.embeddings = torch.Tensor(embeddings)
        self.data = data

    def __len__(self):
        return int(self.data.shape[0])
    
    def __getitem__(self, index):
        return self.data[index], self.embeddings[index]
    

class default_encoder(nn.Module):
    
    def __init__(self, dims, n_nodes = 100, n_components = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(dims), n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_components),
        )

        if ACCEL == 'gpu':
            self.encoder = self.encoder.cuda()

    
    def forward(self, X):
        return self.encoder(X)


class default_decoder(nn.Module):
    
    def __init__(self, dims, n_nodes = 100, n_components = 2):
        super().__init__()
        self.dims = dims
        self.decoder = nn.Sequential(
            nn.Linear(n_components, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, n_nodes),
            nn.ReLU(),
            nn.Linear(n_nodes, np.prod(dims)),
        )

        if ACCEL == 'gpu':
            self.decoder = self.decoder.cuda()

    
    def forward(self, X):
        return self.decoder(X).view(X.shape[0], *self.dims)


def convert_distance_to_probability(distances, a = 1.0, b = 1.0):
    return -torch.log1p(a * distances ** (2 * b))


def compute_cross_entropy(
    probabilities_graph, probabilities_distance, epsilon=1e-4, repulsion_strength=1.0
):
    # cross entropy
    attraction_term = -probabilities_graph * torch.nn.functional.logsigmoid(
        probabilities_distance
    )
    repellant_term = (
        - (1.0 - probabilities_graph) * (
            torch.nn.functional.logsigmoid(probabilities_distance)
            - probabilities_distance
        ) * repulsion_strength
    )

    # balance the expected losses between atrraction and repel
    CE = attraction_term + repellant_term
    return attraction_term, repellant_term, CE


def umap_loss(embedding_to, embedding_from, _a, _b, batch_size, negative_sample_rate = 5):

    # get negative samples by randomly shuffling the batch
    embedding_neg_to = embedding_to.repeat(negative_sample_rate, 1)
    repeat_neg = embedding_from.repeat(negative_sample_rate, 1)
    embedding_neg_from = repeat_neg[torch.randperm(repeat_neg.shape[0])]
    distance_embedding = torch.cat(
        (
            (embedding_to - embedding_from).norm(dim=1),
            (embedding_neg_to - embedding_neg_from).norm(dim=1),
        ),
        dim=0,
    )

    # convert probabilities to distances
    probabilities_distance = convert_distance_to_probability(distance_embedding, _a, _b)
    # set true probabilities based on negative sampling
    probabilities_graph = torch.cat(
        (torch.ones(batch_size),
        torch.zeros(batch_size * negative_sample_rate)), dim=0,
    )

    # compute cross entropy
    (attraction_loss, repellant_loss, ce_loss) = compute_cross_entropy(
        probabilities_graph.cuda() if ACCEL == 'gpu' else probabilities_graph,
        probabilities_distance.cuda() if ACCEL == 'gpu' else probabilities_distance,
    )

    loss = torch.mean(ce_loss)
    return loss


def get_umap_graph(x, n_neighbors = 10, metric = "cosine", random_state = None):

    random_state = check_random_state(None) if random_state == None else random_state
    # number of trees in random projection forest
    n_trees = 5 + int(round((x.shape[0]) ** 0.5 / 20.0))
    # max number of nearest neighbor iters to perform
    n_iters = max(5, int(round(np.log2(x.shape[0]))))
    # distance metric

    # get nearest neighbors
    nnd = NNDescent(
        x.reshape((len(x), np.prod(np.shape(x)[1:]))),
        n_neighbors=n_neighbors,
        metric=metric,
        n_trees=n_trees,
        n_iters=n_iters,
        max_candidates=60,
        verbose=True,
    )

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # get indices and distances
    knn_indices, knn_dists = nnd.neighbor_graph

    # build fuzzy_simplicial_set
    umap_graph, sigmas, rhos = fuzzy_simplicial_set(
        X=x,
        n_neighbors=n_neighbors,
        metric=metric,
        random_state=random_state,
        knn_indices=knn_indices,
        knn_dists=knn_dists,
    )

    return umap_graph


class umap_model(pl.LightningModule):

    def __init__(
        self, lr: float,
        encoder: nn.Module, decoder = None,
        beta = 1.0, min_dist = 0.1,
        reconstruction_loss = F.binary_cross_entropy_with_logits,
        landmark_data = None, landmark_embedding = None, landmark_weight = 0.01,
        disable_umap_loss = False
    ):
        super().__init__()
        self.lr = lr
        self.encoder = encoder
        self.decoder = decoder
        self.beta = beta  # weight for reconstruction loss

        self.landmark_weight = landmark_weight
        self.use_landmark = (landmark_data is not None) and (landmark_embedding is not None)
        
        if self.use_landmark:
            self.landmark_data = torch.tensor(landmark_data, dtype = torch.float32)
            self.landmark_embedding = torch.tensor(landmark_embedding, dtype = torch.float32)
            if ACCEL == 'gpu':
                self.landmark_data = self.landmark_data.cuda()
                self.landmark_embedding = self.landmark_embedding.cuda()

        self.reconstruction_loss = reconstruction_loss
        self._a, self._b = find_ab_params(1.0, min_dist)
        self.disable_umap_loss = disable_umap_loss


    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr = self.lr)


    def training_step(self, batch, batch_idx):

        if not self.disable_umap_loss:
            
            (edges_to_exp, edges_from_exp) = batch
            embedding_to, embedding_from = \
                self.encoder(edges_to_exp), \
                self.encoder(edges_from_exp)
        
            encoder_loss = umap_loss(
                embedding_to,
                embedding_from,
                self._a,
                self._b,
                edges_to_exp.shape[0],
                negative_sample_rate = 5,
            )

            self.log("umap_loss", encoder_loss, prog_bar=True)

            if self.use_landmark:
                selection = torch.randint(0, self.landmark_data.shape[0], (edges_to_exp.shape[0],))
                embedding_parametric = self.encoder(self.landmark_data[selection,:])
                landmark_loss = mse_loss(embedding_parametric, self.landmark_embedding[selection,:])
                self.log("landmark_loss", landmark_loss, prog_bar=True)
                encoder_loss += self.landmark_weight * landmark_loss

            if self.decoder:
                recon = self.decoder(embedding_to)
                recon_loss = self.reconstruction_loss(recon, edges_to_exp)
                self.log("recon_loss", recon_loss, prog_bar=True)
                return encoder_loss + self.beta * recon_loss

            else: return encoder_loss
        
        else:

            data, embedding = batch
            embedding_parametric = self.encoder(data)
            encoder_loss = mse_loss(embedding_parametric, embedding)
            self.log("encoder_loss", encoder_loss, prog_bar = True)

            if self.decoder:
                recon = self.decoder(embedding_parametric)
                recon_loss = self.reconstruction_loss(recon, data)
                self.log("recon_loss", recon_loss, prog_bar = True)
                return encoder_loss + self.beta * recon_loss
            
            else: return encoder_loss


class data_module(pl.LightningDataModule):
    
    def __init__(self, dataset, batch_size, num_workers):

        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers


    def train_dataloader(self) -> DataLoader:

        return DataLoader(
            dataset = self.dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            shuffle = True,
        )


class pumap:

    def __init__(
        self,
        encoder = None, decoder = None,
        n_neighbors = 10, min_dist = 0.1,
        metric = "euclidean", n_components = 2, beta = 1.0,
        reconstruction_loss = F.binary_cross_entropy_with_logits,
        random_state = 42, n_nodes = 100,
        lr = 1e-3, epochs = 10, batch_size = 64, 
        num_workers = 1, num_gpus = 1
    ):
        
        self.encoder = encoder
        self.decoder = decoder
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.n_components = n_components
        self.beta = beta
        self.reconstruction_loss = reconstruction_loss
        self.random_state = np.random.RandomState(random_state)
        self.random_state_int = random_state
        self.lr = lr
        self.epochs = epochs
        self.n_nodes = n_nodes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_gpus = num_gpus

        self.model = umap_model(
            self.lr,
            encoder,
            decoder,
            beta = self.beta,
            min_dist = self.min_dist,
            reconstruction_loss = self.reconstruction_loss,
        )
    
    def fit(self, X, init_pos = True):

        trainer = pl.Trainer(accelerator = ACCEL, devices = 'auto', max_epochs = self.epochs)
        encoder = (
            default_encoder(X.shape[1:], n_nodes = self.n_nodes, n_components = self.n_components)
            if self.encoder is None
            else self.encoder
        )

        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)


        graph = get_umap_graph(
            X,
            n_neighbors = self.n_neighbors,
            metric = self.metric,
            random_state = self.random_state,
        )

        if init_pos:
            
            embedding = spectral_layout(
                X,
                graph,
                self.n_components,
                self.random_state,
                metric = self.metric
            )

            # add a little noise to avoid local minima for optimization to come
            embedding = noisy_scale_coords(
                embedding, self.random_state, max_coord = 10, noise = 0.0001
            )

            self.fit_layout(X, embedding)

        self.model = umap_model(
            self.lr,
            encoder,
            decoder,
            beta = self.beta,
            min_dist = self.min_dist,
            reconstruction_loss = self.reconstruction_loss,
        )

        trainer.fit(
            model = self.model,
            datamodule = data_module(
                umap_dataset(X, graph), self.batch_size, self.num_workers
            ),
        )

    
    def fit_layout(self, X, embeddings):

        trainer = pl.Trainer(accelerator = ACCEL, devices = 'auto', max_epochs = self.epochs)
        encoder = (
            default_encoder(X.shape[1:], n_nodes = self.n_nodes, n_components = self.n_components)
            if self.encoder is None
            else self.encoder
        )

        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)

        layout = umap_model(
            self.lr,
            encoder,
            decoder,
            beta = self.beta,
            min_dist = self.min_dist,
            reconstruction_loss = self.reconstruction_loss,
            disable_umap_loss = True
        )

        trainer.fit(
            model = layout,
            datamodule = data_module(
                layout_dataset(X, embeddings), self.batch_size, self.num_workers
            ),
        )
    

    def fit_with_landmark(self, X, landmarks = None, landmark_weight = 0.05):

        trainer = pl.Trainer(accelerator = ACCEL, devices = 'auto', max_epochs = self.epochs)
        encoder = (
            default_encoder(X.shape[1:], n_nodes = self.n_nodes, n_components = self.n_components)
            if self.encoder is None
            else self.encoder
        )

        if self.decoder is None or isinstance(self.decoder, nn.Module):
            decoder = self.decoder
        elif self.decoder == True:
            decoder = default_decoder(X.shape[1:], self.n_components)

        not_landmark = np.isnan(np.sum(landmarks, axis = 1))
        landmark_data = X[~ not_landmark, :]
        landmark_embedding = landmarks[~ not_landmark, :]

        self.model = umap_model(
            self.lr, encoder, decoder,
            beta = self.beta,
            reconstruction_loss = self.reconstruction_loss,
            landmark_data = landmark_data,
            landmark_embedding = landmark_embedding,
            landmark_weight = landmark_weight
        )

        graph = get_umap_graph(
            X,
            n_neighbors = self.n_neighbors,
            metric = self.metric,
            random_state = self.random_state,
        )

        trainer.fit(
            model = self.model,
            datamodule = data_module(
                umap_dataset(
                    X, graph, landmark_data = landmark_data, 
                    landmark_embedding = landmark_embedding
                ),
                self.batch_size,
                self.num_workers
            )
        )

    @torch.no_grad()
    def transform(self, X):
        x = torch.tensor(X, dtype = torch.float32)
        if ACCEL == 'gpu' and next(self.model.encoder.parameters()).is_cuda: 
            x = x.cuda()
        return self.model.encoder(x).detach().cpu().numpy()

    @torch.no_grad()
    def inverse_transform(self, Z):
        x = torch.tensor(Z, dtype = torch.float32)
        if ACCEL == 'gpu' and next(self.model.encoder.parameters()).is_cuda: 
            x = x.cuda()
        return self.model.decoder(x).detach().cpu().numpy()

    def save(self, path):
        if self.model.encoder is not None:
            torch.save(self.model.encoder, os.path.join(path, 'encoder.pt'))
        if self.model.decoder is not None:
            torch.save(self.model.decoder, os.path.join(path, 'decoder.pt'))
        
        model_params = {
            'n_neighbors': self.n_neighbors, 
            'min_dist': self.min_dist,
            'metric': self.metric, 
            'n_components': self.n_components, 
            'beta': self.beta,
            'random_state': self.random_state_int, 
            'n_nodes': self.n_nodes,
            'lr': self.lr, 
            'epochs': self.epochs, 
            'batch_size': self.batch_size, 
            'num_workers': self.num_workers,
            'num_gpus': self.num_gpus
        }

        with open(os.path.join(path, 'configs.pkl'), 'wb') as f:
            pickle.dump(model_params, f)


def load_pumap(path, cpu_only = True):
    encoder = None
    decoder = None
    
    if os.path.exists(os.path.join(path, 'encoder.pt')):
        print('[*] reading encoder ...')
        encoder = torch.load(
            os.path.join(path, 'encoder.pt'), weights_only = False,
            map_location = torch.device('cuda') if ACCEL == 'gpu' and not cpu_only \
                else torch.device('cpu')
        )

        if ACCEL == 'gpu' and not cpu_only: 
            encoder = encoder.cuda()

    if os.path.exists(os.path.join(path, 'decoder.pt')):
        print('[*] reading decoder ...')
        decoder = torch.load(
            os.path.join(path, 'decoder.pt'), weights_only = False,
            map_location = torch.device('cuda') if ACCEL == 'gpu' and not cpu_only \
                else torch.device('cpu')
        )

        if ACCEL == 'gpu' and not cpu_only: 
            decoder = decoder.cuda()

    with open(os.path.join(path, 'configs.pkl'), 'rb') as f:
        params = pickle.load(f)
    
    return pumap(encoder, decoder, **params)

