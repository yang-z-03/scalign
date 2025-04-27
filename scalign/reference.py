
import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import importlib

import pickle
import pandas as pd

import matplotlib as mpl
from matplotlib import cm
import matplotlib.patches as mpatches
import matplotlib.text as mtext
    
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import matplotlib.patheffects as mpe
from matplotlib.colors import ListedColormap as listedcm

import torch
from scalign.encoders import default_encoder

# Set default font family to 'sans-serif'
plt.rcParams['font.family'] = 'sans-serif'
# Specify a list of sans-serif fonts to try
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans', 'Verdana']

ftprop = fm.FontProperties(fname = '/home/yang-z/.local/share/fonts/Arial/arial.ttf')
ftboldprop = fm.FontProperties(fname = '/home/yang-z/.local/share/fonts/Arial/arial-b.ttf', weight = 'bold')


class index_object:
    pass


class index_object_handler:

    def __init__(self):
        self.index = 0
        self.text = ''
        self.color = 'black'

    def set_option(self, index, text, color):
        self.index = index
        self.text = text
        self.color = color
    
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        
        center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
        patch = mpatches.Ellipse(
            xy = center, 
            width = (height + y0) * 1.5,
            height = (height + y0) * 1.5,
            color = self.color
        )

        annot = mtext.Text(
            x = center[0], y = center[1] - (height + y0) * 0.1, text = str(self.index), color = 'black',
            va = 'center', ha = 'center', fontproperties = ftboldprop,
            transform = handlebox.get_transform(), size = 8
        )
        
        annot.set_path_effects([
            mpe.Stroke(linewidth = 2, foreground = 'white'),
            mpe.Normal()
        ])
        
        handlebox.add_artist(patch)
        handlebox.add_artist(annot)
        return patch


class reference:
    """
    Reference atlas

    This is the core export class of ``scalign``. It loads the reference atlas from a directory.
    At present, this package do not contain methods to building a reference dump automatically,
    this will be added in later versions. 

    Attributes
    ----------

    path : str
        The directory to the reference atlas. This should always contains a ``metadata.h5ad``
        file and a ``scvi`` directory, and contain either or both ``embedder.pkl`` and / or 
        ``parametric`` directory. These two store the non-parametric and parametric UMAP embedder
        respectively. Parametric UMAP embedder requires ``keras >= 3.1`` and ``tensorflow >= 2.0``
        as additional dependencies, and can run much faster if you have configured valid GPUs.
        The non-parametric UMAP embedder serves as a fallback point and runs faster than the
        parametric model when no GPU is installed.

    key_atlas_var : str
        The matching gene metadata in the atlas to the query set. You may pick an identifier that
        your query set contains. It is ``.ensembl`` by default indicating a column of ENSEMBL IDs
        you should set ``query(key_var = '...')`` to the corresponding ENSEMBL IDs as this.

    use_parametric_if_available : bool
        If set to ``True``, this will use parametric model if tensorflow is installed.
        If set to ``False``, you may force the aligner to use the non-parametric one.
    
    use_expression_if_available : str
        If the expression data of the atlas is available, try to load them into the model. One can
        build a reference atlas with gene expression quantification within them by supplying a 
        log-normalized matrix. This will enable more analysis and visualization capacity of the
        atlas mapper. However, if the atlas is relatively large, this make take extra long time to
        load and more disk space (as well as working memory). For a lite distribution of the atlas,
        one do not need the expression data, and the mapping program takes an average of 5 Gb memory
        to perform its job for a 1,250,000 cell atlas. (1.25 M cells) This is considered a large
        atlas already, but is capable to analysis on a single laptop computer. However, the expression
        matrix of atlas at such size may become at least ~150 Gb. A full distribution that contains 
        such data should take about 160 Gb disk space, and nearly 200 Gb memory to load them
        successfully. So the user should check the configuration of their machine before turning
        the switch on. Otherwise it will crash the program.
    """

    # load the reference dataset from path.
    def __init__(
        self, path,
        key_atlas_var = '.ensembl',
        use_parametric_if_available = True,
        use_expression_if_available = False,
        use_gpu_if_available = True
    ):

        import scanpy
        
        self.error = False
        self._has_embedder_np = True
        self._has_embedder_p = True
        self._use_parametric = False
        self._has_expression = False
        
        f_scvi = os.path.join(path, 'scvi', 'model.pt')
        f_meta = os.path.join(path, 'metadata.h5ad')
        f_emb = os.path.join(path, 'umap')
        f_param = os.path.join(path, 'parametric')
        f_expr = os.path.join(path, 'expression.h5ad')

        if not os.path.exists(f_scvi):
            print(f'[!] scvi model not found. at {f_scvi}')
            self.error = True
            return

        if not os.path.exists(f_meta):
            print(f'[!] reference dataset not found. at {f_meta}')
            self.error = True
            return

        if not os.path.exists(f_emb):
            print(f'[!] this reference set do not have a non-parametric umap embedder.')
            self._has_embedder_np = False

        if not os.path.exists(f_param):
            print(f'[!] this reference set do not have a parametric umap embedder.')
            self._has_embedder_p = False

        if not (self._has_embedder_p or self._has_embedder_np):
            print(f'[!] no embedder found.')
            self.error = True
            return
        
        # we will use the parametric embedder if there is one.
        else:
            if use_parametric_if_available: 
                self._use_parametric = self._has_embedder_p
            else: self._use_parametric = False

        # read the atlas metadata.
        self._metadata = scanpy.read(f_meta)

        # load the umap transformer
        if self._use_parametric:
            try:
                from scalign.parametric import load_pumap
                self._embedder = load_pumap(
                    os.path.join(path, 'parametric'),
                    cpu_only = not use_gpu_if_available
                )

                embeddings = np.load(os.path.join(path, 'parametric', 'embeddings.npy'))
                self._metadata.obsm[self._metadata.uns['parametric']['precomputed']] = embeddings
                
            except Exception as e:
                print(f'[!] your environment is not available to load parametric model.')
                print(f'[!] will fallback to non-parametric model. ')
                print(e)
                self._use_parametric = False
                if not self._has_embedder_np:
                    print(f'[!] fallback non-parametric model do not exist.')
                    self.error = True
                    return
        
        if not self._use_parametric:
            with open(os.path.join(path, 'umap', 'embedder.pkl'), 'rb') as f:
                self._embedder = pickle.load(f)
            embeddings = np.load(os.path.join(path, 'umap', 'embeddings.npy'))
            self._metadata.obsm[self._metadata.uns['umap']['precomputed']] = embeddings

        if os.path.exists(f_expr) and use_expression_if_available:
            print(f'[*] loading the expression matrix into memory ...')
            self._expression = scanpy.read(f_expr)
            if self._expression.n_obs != self._metadata.n_obs or \
                self._expression.n_vars != self._metadata.n_vars:
                print(f'[!] the size of metadata and expression matrix do not match.')
                print(f'[!] you should check if the atlas installation is corrupted.')
                self.error = True
                del self._expression
                return
            
            self._has_expression = True

        self.directory = path
        self.scvi = os.path.join(path, 'scvi')

        # check mandatory metadata
        print(f'[*] load reference atlas of size [obs] {self._metadata.n_obs} * [var] {self._metadata.n_vars}')
        assert 'latent' in self._metadata.uns.keys()
        assert 'precomputed' in self._metadata.uns['latent'].keys()
        assert 'latent' in self._metadata.uns['latent'].keys()
        assert self._metadata.uns['latent']['precomputed'] in self._metadata.obsm.keys()
        assert self._metadata.obsm[self._metadata.uns['latent']['precomputed']].shape[1] == \
            self._metadata.uns['latent']['latent']
        
        if self._use_parametric:
            assert 'parametric' in self._metadata.uns.keys()
            assert 'landmark' in self._metadata.uns.keys()
            assert 'precomputed' in self._metadata.uns['parametric'].keys()
            assert 'latent' in self._metadata.uns['parametric'].keys()
            assert 'dimension' in self._metadata.uns['parametric'].keys()
            assert self._metadata.uns['parametric']['precomputed'] in self._metadata.obsm.keys()
            assert self._metadata.obsm[self._metadata.uns['parametric']['precomputed']].shape[1] == \
                self._metadata.uns['parametric']['dimension']
            assert self._metadata.uns['parametric']['latent'] == self._metadata.uns['latent']['latent']
        else:
            assert 'umap' in self._metadata.uns.keys()
            assert 'precomputed' in self._metadata.uns['umap'].keys()
            assert 'latent' in self._metadata.uns['umap'].keys()
            assert 'dimension' in self._metadata.uns['umap'].keys()
            assert self._metadata.uns['umap']['precomputed'] in self._metadata.obsm.keys()
            assert self._metadata.obsm[self._metadata.uns['umap']['precomputed']].shape[1] == \
                self._metadata.uns['umap']['dimension']
            assert self._metadata.uns['umap']['latent'] == self._metadata.uns['latent']['latent']

        # we should manually ensure the n and k is both unduplicated.
        assert key_atlas_var in self._metadata.var.keys()
        var_keys = self._metadata.var[key_atlas_var].tolist()
        var_names = self._metadata.var_names.tolist()
        converter = {}
        for n, k in zip(var_names, var_keys):
            converter[k] = n
        self._converter = converter

        pass


    @property
    def observations(self):
        """
        Readonly observation metadata of the atlas
        """
        return self._metadata.obs


    @property
    def variables(self):
        """
        Readonly variable metadata of the atlas
        """
        return self._metadata.var
    
    
    @property
    def use_parametric(self):
        """
        Whether the atlas mapper use the parametric UMAP model. You may alter the 
        ``use_parametric_if_available`` switch when creating the reference to inform that you prefer
        a parametric model to be used. However, if you did not configure ``keras`` and ``tensorflow``
        packages correctly, the program will automatically fallback to non-parametric model as a 
        default. The actual model of use might not be what you expected, as you can check this
        property to see which model is actually loaded and used.
        """
        return self._use_parametric
    

    @property
    def use_expression(self):
        """
        Whether the atlas mapper load the expression matrix.
        """
        return self._has_expression


    @property
    def use_gpu(self):
        """
        Whether the atlas mapper load the expression matrix.
        """
        if self.use_parametric:
            return next(self._embedder.model.encoder.parameters()).is_cuda
        else: return False
    

    @property
    def expression(self):
        """
        Get the expression matrix in log normalized counts.
        """
        if self.use_expression:
            return self._expression.X
        else: 
            print(f'[!] the model is not loaded with an expression matrix')
            return None
    
    
    @property
    def converter(self):
        """
        The converter dictionary from the key specified in ``key_atlas_var`` corresponding in the
        variable metadata to the atlas variable key. 
        """
        return self._converter
    

    @property
    def epoch(self):
        """
        Number of epochs for each sample that came across when training.
        """
        return self._embedder.n_epochs
    
    @epoch.setter
    def epoch(self, value):
        self._embedder.n_epochs = value


    @property
    def reproduceable(self):
        """
        Whether the model is trained with a deterministic random state. Setting the random state
        will make the training process reproduceable, but can only use 1 CPU cores during neighbor
        finding. This random seed can not be changed after the model is built. So this field is
        readonly. You can make a request to the atlas distributor if you would like a reproduceable model,
        or to train a model with counts by yourself.

        However, since the prediction process do not alter the model, you will still get reproduceable
        results if you map the query dataset with ``retrain`` set to ``False``. Minor differences will
        only occur if you retrain the model, and this won't change much since we apply an additional
        loss trying to keep the atlas of the same shape.
        """
        return isinstance(self._embedder.random_state, int)

    
    def network_summary(self):
        """
        Print the network summary for a parametric model. This function will print an error text
        if the model is loaded with non-parametric model.
        """
        if self.use_parametric:
            try:
                import torchinfo
                torchinfo.summary(self._embedder.encoder)
            except ImportError:
                print('[!] summary needs torchinfo installed.')
        else: print('[!] the atlas embedder is non-parametric.')

    
    def training_loss(self):
        """
        Get the training loss vector that recorded during the model's training and retraining
        process. The original vector 
        """
        if self.use_parametric:
            return self._embedder._history['loss']
        else: print('[!] the atlas embedder is non-parametric.')

    
    def __str__(self):
        if self.error:
            return 'Loading failed. Check your installation.'
        else:
            return '\n'.join([
                f'An atlas of size [obs] {self._metadata.n_obs} * [var] {self._metadata.n_vars}. \n',
                f'              [use parametric]: {self.use_parametric}',
                f'    [expression matrix loaded]: {self._has_expression}',
                f'      [parametric model found]: {self._has_embedder_p}',
                f'  [non-parametric model found]: {self._has_embedder_np}',
                f'           [scvi latent space]: {self._metadata.uns["latent"]["latent"]}',
                f'             [gpu accelerated]: {self.use_gpu}'
            ])


    def query(
        self, input, 
        batch_key = None, 
        key_var = None,
        key_query_latent = 'scvi',
        key_query_embeddings = 'umap',
        scvi_epoch_reduction = 3,
        retrain = False,
        landmark_reduction = 60,
        landmark_loss_weight = 0.01,
        n_jobs = 1, n_epochs = 10
    ):
        """
        Query the reference atlas with a dataset
        
        Parameters
        ----------

        input : anndata.AnnData
            The query dataset to be aligned. The variable identifier will be mapped to the reference
            atlas by the specified variable metadata column (in ``reference(key_atlas_var = ...)``).
            This column in the atlas metadata of genes will match the query dataset's metadata column
            specified by ``key_var``. If ``key_var`` is not specified, the query dataset's variable names
            will be used as identifier.

            The query dataset **must** have unique variable names and observation names. Otherwise
            the program will raise an error. You can use ``index.is_unique`` to check this.

        batch_key : str
            The observation metadata key specifying sample batches. This will be used to correct
            batch effect using ``scvi`` model. If not specified, the program will generate a obs
            slot named ``batch`` and assign all samples to the same batch. Note that if you have
            an observation metadata column named ``batch``, it will be overwritten.

        key_var : str
            The variable metadata key specifying the gene names. This should match the key selected
            in the atlas (by default, a list of ENSEMBL IDs). If not specified, the program will use
            the variable names. You should make sure that the contents in this column are unique.
            After the alignment, the variable names will be transformed to the same as the atlas.
            The original variable names will be stored in ``.index`` slot. You should keep a copy of
            that if you need them thereafter.
        
        key_query_latent : str
            The obsm key to store scVI latent space. If there is already a key with the same name,
            the calculation of scVI components will skip, and the data inside the slot will be used
            directly as the scVI latent space.
        
        key_query_embeddings : str
            The obsm key to store UMAP embeddings. This embeddings will *mostly* share the same 
            structure as the reference atlas. Since the exact UMAP model is used to transform the
            latent space. If ``retrain`` is set to ``False``, the UMAP will just serve as a prediction
            model to transform between dimensions without training on them. This is rather fast,
            but may introduce errors in the predicted embeddings (since the model have not seen
            the data totally during its training). Non-parametric model do not support retraining,
            and can only be used as a prediction model. 

            Parametric UMAP models have the capability to be retrained with new data. This will help
            the new data points better integrated into the atlas, and revealing more accurate
            alignment. However, the atlas embedding is somewhat affected by the new ones. Though we
            use landmarking points to help preserve the original structure, there may be some 
            small differences between the new atlas and the original one.

            If there is already a key with the same name, UMAP embedding calculation will be skipped.

        scvi_epoch_reduction : int
            Since the scVI model has been trained, we just need a few epochs to adapt it to the new
            data. The epochs may be less than what scVI expected to be. This saves a lot of time when
            running on CPU machines without reducing the performance too much. By default, the 
            reduction ratio is set to 4.
        
        retrain : bool
            Whether to retrain the model. This is only supported for parametric model.
        
        landmark_reduction : int
            Partition to randomly select as landmarking points. The trainer will select 1 out of 
            N points from the original atlas to help make the overall space not change dramatically.
            The less the reduction ratio is, the more samples from the original atlas will be 
            used in retraining. By default is set to 60.
        
        landmark_loss_weight : float
            The weight of the landmark loss. By default 0.01.

        n_jobs : int
            Number of threads to use when running UMAP embedding.

        Returns
        -------
        anndata.AnnData

            The modified anndata object. with the following slots set:
            
            * ``.obs``: ``batch``
            * ``.var``: ``index``, ``var_names``
            * ``.obsm``: ``key_query_latent``, ``key_query_embeddings``
            * ``.uns``: ``.align``

            These changes is made inplace, however, the modified object is still
            returned for convenience.
        """

        import scvi
        import anndata

        self._embedder.n_jobs = n_jobs
        query = anndata.AnnData(X = input.X.copy())

        assert query.var.index.is_unique
        assert query.obs.index.is_unique

        # the scvi requires obs[batch] and var names to be .ugene.
        # so we transform to adapt to it.
        
        query.var['index'] = query.var_names.tolist()
        if batch_key is not None:
            assert batch_key in input.obs.keys()
            query.obs['batch'] = input.obs[batch_key].tolist()
        else: 
            print(f'[!] do not supply the batch key. assume they all came from the same batch.')
            query.obs['batch'] = 'whole'
            
        if key_var is not None:
            assert key_var in input.var.keys()
            qkey = input.var[key_var].tolist()
        else: qkey = input.var_names.tolist()
        qindex = input.var_names.tolist()

        if key_query_latent in input.obsm.keys():
            query.obsm[key_query_latent] = input.obsm[key_query_latent].copy()
        
        if key_query_embeddings in input.obsm.keys():
            query.obsm[key_query_embeddings] = input.obsm[key_query_embeddings].copy()

        qconv = []
        n_nan = 0
        # the qindex must not contain nan's
        for x, idx in zip(qkey, qindex):
            if x in self.converter.keys(): qconv.append(self.converter[x])
            else: 
                qconv.append(idx)
                n_nan += 1

        if n_nan > 0:
            print(f'[!] {n_nan} nan values inside the selected column [{key_var}].')
            print(f'[!] impute the nan with the index column.')
        
        query.var_names = qconv

        if key_query_latent not in query.obsm.keys():
            
            print(f'[*] preparing query data ...')
            scvi.model.SCVI.prepare_query_anndata(query, self.scvi)
            print(f'[*] constructing query model for batch correction ...')
            query_model = scvi.model.SCVI.load_query_data(query, self.scvi)
        
            max_epochs_scvi = np.min([round((20000 / query.n_obs) * 400), 400]) // scvi_epoch_reduction
            print(f'[*] will automatically train {max_epochs_scvi} epochs ...')
            query_model.train(max_epochs = int(max_epochs_scvi), plan_kwargs = { 'weight_decay': 0.0 })
            query.obsm[key_query_latent] = query_model.get_latent_representation()

        else: print(f'[>] skipped calculation of scvi, since it already exist.')
            
        query_latent = query.obsm[key_query_latent]
        
        if self.use_parametric:
            if retrain:
                
                print(f'[>] retraining umap embedding ...')
                latent = self._metadata.obsm[self._metadata.uns['latent']['precomputed']]
                embeddings = self._metadata.obsm[self._metadata.uns['parametric']['precomputed']]
                landmark_idx = self._metadata.uns['landmark'][
                    0 : (len(self._metadata.uns['landmark']) // landmark_reduction)
                ]
                
                # append the landmark points
                finetune = np.concatenate((query_latent, latent[landmark_idx]))
                
                # landmarks vector, which is nan where we have no landmark information.
                landmarks = np.stack(
                    [np.array([np.nan, np.nan])] * query_latent.shape[0] + 
                    list(embeddings[landmark_idx])
                )
                
                # set landmark loss weight and continue training our parametric umap model.
                self._embedder.epochs = n_epochs
                self._embedder.fit_with_landmark(
                    finetune, landmarks = landmarks, landmark_weight = landmark_loss_weight
                )

                print(f'[*] umap transforming in atlas latent space ...')
                retrain_atlas = self._embedder.transform(latent)
                self._metadata.obsm[self._metadata.uns['parametric']['precomputed']] = retrain_atlas

        if (not self.use_parametric) and retrain:
            print(f'[!] non-parametric umap do not support retraining!')
        
        if key_query_embeddings not in query.obsm.keys():
            print(f'[*] umap transforming querying dataset ...')
            query_embeddings = self._embedder.transform(query_latent)
            query.obsm[key_query_embeddings] = query_embeddings
        
        else: print(f'[>] skipped calculation of umap, since it already exist.')

        # remove autogenerated metadata by scvi.
        if '_scvi_manager_uuid' in query.uns.keys(): del query.uns['_scvi_manager_uuid']
        if '_scvi_uuid' in query.uns.keys(): del query.uns['_scvi_uuid']
        if '_scvi_batch' in query.obs.keys(): del query.obs['_scvi_batch']
        if '_scvi_labels' in query.obs.keys(): del query.obs['_scvi_labels']
        
        query.uns['.align'] = {
            'retrain': retrain,
            'parametric': self.use_parametric
        }

        # a fix to the illegal var name
        # the var names may contain nan's if the given index key contains nan.
        # 
        # when we prepare a dataset, the scvi routines will automatically align the query
        # set's index onto the reference, and assign the variable names the same as atlas
        # e.g. g1 ~ g55000 etc. and by this operation, the genes that do not present in
        # the atlas is removed and genes that do not exist in the query set is inserted
        # this will introduce nan's into the columns other that the index.
        #
        # when we next feed them in, the nan's will cause problems. i try to fix them by
        # substitute the missing values with values inside the index before transformation.
        
        query.var_names = [str(x) for x in query.var_names.tolist()]
        return query

    
    def density(
        self, query, 
        stratification = 'query',

        # atlas
        atlas_ptsize = 2,

        atlas_embedding = None,
        atlas_color_mode = 'categorical',
        key_atlas_var = '.name',
        atlas_gene = None,

        atlas_hue = None,
        atlas_hue_order = None,
        atlas_default_color = '#e0e0e0',
        atlas_alpha = 1.0,
        atlas_palette = 'hls',
        atlas_rasterize = True,
        
        atlas_annotate = True,
        atlas_annotate_style = 'index',
        atlas_annotate_foreground = 'black',
        atlas_annotate_stroke = 'white',
        atlas_legend = True,

        # query plotting options
        key_query_embeddings = 'umap',
        query_plot = True,
        query_ptsize = 8, 
        query_hue = None,
        query_hue_order = None,
        query_default_color = 'black',
        query_alpha = 0.5,
        query_palette = 'hls',
        query_rasterize = True,

        query_annotate = True,
        query_annotate_style = 'index',
        query_annotate_foreground = 'black',
        query_annotate_stroke = 'white',
        query_legend = True,

        # contour plotting option.
        contour_plot = True,
        contour_fill = False,
        contour_hue = None,
        contour_hue_order = None,
        contour_linewidth = 0.8,
        contour_default_color = 'black',
        contour_palette = 'hls',
        contour_alpha = 1,
        contour_levels = 10,
        contour_bw = 0.5,

        legend_col = 1,
        add_outline = False,
        outline_color = 'black',
        width = 5, height = 5, dpi = 100, elegant = False,
        title = 'Embeddings', save = None
    ):
        """
        Plot mapping density

        This function is a helper to plot alignment density. Either be shown to the interactive
        console, or save to disk files.

        Parameters
        ----------

        query : anndata.AnnData
            The mapped query set. Must run with ``reference.query()`` beforehand. Since this function
            requires the data to contain ``.uns['.align']`` and ``.obsm['umap']``.
        
        stratification : Literal['query', 'atlas'] = 'query'
            The plot function will only show one in the two cases. Either coloring a categorical 
            metadata from the atlas, or a metadata from the query set. The legend will automatically
            show for each.
        
        add_outline : bool = False
            Whether to add an outline to the atlas embedding region. This may stress the atlas boundary.
        
        outline_color : str = 'black'
            A named matplotlib color (or hex code) to the outline

        query_plot : bool = True
            Whether to plot the scatter points from the query dataset. Note that this do not affect
            the plotting of query labels or query legends if they are set to be plotted.

        contour_plot : bool = True
            Whether to plot the isoheight contours.
        
        legend_col : int = 1
            Number of columns to display legend markers. Set to an adequate number for aethesty
            when the groupings have a lot of possible values.
        
        atlas_color_mode : str = Literal['categorical', 'expression']
            How to plot the atlas color. If set to ``categorical``, this will require ``atlas_hue``
            to set to a categorical metadata name. If set to ``expression``, this will plot the 
            expression levels of a specified gene (with ``atlas_gene``) on the base UMAP. This requires
            an expression matrix to be loaded into the atlas when creating it (by supplying
            ``use_expression_if_available`` argument)

        atlas_gene : str = None
            The gene to plot. Must be valid name presented in ``.variables[key_atlas_var]``.
        
        ptsize : float = (atlas: 2, query: 8)
            The point size of the atlas basis plot and the query scatter. Typically the query data
            points should be plotted larger than the atlas, since the atlas contains more cells.
        
        hue : str = None
            The categorical variable specified for groupings of the atlas or the query. Note that 
            only the selected layer by ``stratification`` will be plotted, since plotting both
            the data with colors will obfuscate the graph. This variable must exist within the
            ``obs`` slot of the corresponding anndata. If set to `None`, we will plot the data
            points in the same color specified by ``atlas_default_color`` or ``query_default_color``.
        
        order : list[str] = None
            Specify the order of hue variable. This is useful in combination with the manually
            specified palette to determine exact color used.
        
        default_color : str = ('#e0e0e0', 'black')
            A named matplotlib color (or hex code) for the atlas scatter and the query scatter if
            not colored by category. If ``atlas_hue`` or ``query_hue`` is not ``None``, the value
            of this parameter will be ignored, and the coloring of the graph is then specified
            by ``atlas_palette`` and ``query_palette``.
        
        alpha : float = (1, 0.5)
            The transparency of data points.
        
        palette : str | list[str] = 'hls'
            The color palette. Could either be a string indicating named palette names (or following
            the syntax of color palette names by ``seaborn``), or a list of color strings specifying
            exact colors (and their order). If the length of the colors do not meet the length of
            categorical values, the automatic palette cycling rule will be applied by ``matplotlib``.

        rasterize : bool = True
            Whether to rasterize the scatter plot. We strongly recommend setting these values to
            ``True``, for an atlas of a large scale will blow up the graphic object, resulting in
            ridiculously large vector formats and slow performance.
        
        annotate : bool = True
            If a ``hue`` is specified, whether to mark the categories onto the map.
        
        annotate_style : Literal['index', 'label'] = 'index'
            The markers of categories on map. ``index`` will mark a circled index according to the
            legend marker, and ``label`` will mark the category text.
        
        annotate_foreground : str = 'black'
            A named matplotlib color (or hex code). Foreground color to the annotated text.
        
        annotate_stroke : str = 'white'
            A named matplotlib color (or hex code). Stroke color to the annotated text.

        legend : bool = True
            Whether to show the categorical legend.
        
        contour_fill : bool = False
            Whether to fill the isoheight contours with a color gradient. If this is set to ``True``,
            the value of ``contour_linewidth`` will be ignored.

        contour_linewidth : float = 0.8
            The line width of the non-filled isoheight contours.
        
        contour_levels : int | list[float] = 10
            The levels of the contours. If a single integer value is provided, the whole range is
            splitted evenly to match the levels (e.g. setting to ``5`` will have the same effects
            as ``[0.2, 0.4, 0.6, 0.8]``), or specify a list of levels to plot the contour manually.
        
        contour_bw : float = 0.5
            The larger the parameter is, the smoother the contours will be.
        
        width : int = 5
            Width of figure
        
        height : int = 5
            Height of figure
        
        dpi : int = 100
            DPI. If saving to vector graphics (e.g. PDF, SVG etc.), you should note that some part of
            the graphics is rasterized by default to reduce object size. The resolution of such 
            rasterized objects is still affected by DPI.
        
        elegant : int = False
            Show no boundary.

        title : str = 'Embeddings'
            Title of the plot, or ``None`` to hide the title.
        
        save : str = None
            If set to ``None``, the plot will be displayed using ``matplotlib.pyplot.show()``. Otherwise,
            set the parameter to a valid file name to save the image to disk.

        Returns
        -------
        None | Figure
            If ``save`` is ``None``, return the plotting figure in matplotlib format.
            If ``save`` is set, will write the image to disk and return ``None``.
        """

        # we expect a colors.tsv exist under the folder, and specify two columns
        # strat and color. where strat corresponds to the values of the type_annotation
        # column of observation metadata.
        
        do_parametric = query.uns['.align']['parametric']
        slot = (self._metadata.uns['parametric']['precomputed'] if do_parametric else \
            self._metadata.uns['umap']['precomputed']) if atlas_embedding is None else \
            atlas_embedding
        umap_x = self._metadata.obsm[slot][:,0]
        umap_y = self._metadata.obsm[slot][:,1]
        
        umapq_x = query.obsm[key_query_embeddings][:,0]
        umapq_y = query.obsm[key_query_embeddings][:,1]
        sb.set_style('white')
        
        fig, axes = plt.subplots(figsize = (width, height), dpi = dpi)
    
        atlas_data = {
            'x': umap_x, 
            'y': umap_y, 
            'edgecolor': None,
            'legend': False,
            'ax': axes,
            'rasterized': True
        }
    
        if add_outline:
            sb.scatterplot(**atlas_data, color = outline_color, s = atlas_ptsize + 40)
            sb.scatterplot(**atlas_data, color = 'white', s = atlas_ptsize + 20)

        # the atlas scatter plot
        ahue = None
        ahue_order = None
        if atlas_hue is not None and atlas_color_mode == 'categorical':
            ahue = self._metadata.obs[atlas_hue].tolist()
            ahue_order = self._metadata.obs[atlas_hue].value_counts().index.tolist() \
                if atlas_hue_order is None else atlas_hue_order
        
        aexpr = None
        if atlas_color_mode == 'expression':
            if not self.use_expression:
                print(f'[!] this atlas is not loaded with expression data.')
                print(f'[!] there will be no use to set atlas_color_mode = "expression"')
            
            else:
                if not key_atlas_var in self.variables.keys():
                    print(f'[!] {key_atlas_var} does not exist in variable keys')
                elif not atlas_gene in self.variables[key_atlas_var].tolist():
                    print(f'[!] gene {atlas_gene} do not exist in {key_atlas_var}.')
                else:
                    turbo = cm.get_cmap('turbo', 256)
                    cmap = turbo(np.linspace(0, 1, 256))
                    blanked = np.array([0, 0, 0, 1])
                    # np.array([0.95, 0.95, 0.98, 1]) # a tint of bluish gray
                    cmap[:5, :] = blanked
                    cmap = listedcm(cmap)
                    
                    atlas_palette = cmap
                    genes = self.variables[key_atlas_var].tolist()
                    gene_col = genes.index(atlas_gene)
                    aexpr = self.expression[:, gene_col].transpose().todense().tolist()[0]

        atlas_data['rasterized'] = atlas_rasterize
        sb.scatterplot(
            **atlas_data, s = atlas_ptsize,
            alpha = atlas_alpha, palette = atlas_palette, color = atlas_default_color,
            hue = aexpr if aexpr is not None else ahue, hue_order = ahue_order
        )
    
        hue = None
        chue = None
        hue_order = None
        if query_hue is not None:
            hue = query.obs[query_hue].tolist()
            hue_order = query.obs[query_hue].value_counts().index.tolist() \
                if query_hue_order is None else query_hue_order
        
        if contour_hue is not None:
            chue = query.obs[contour_hue].tolist()

        if query_plot:
            sb.scatterplot(
                x = umapq_x, y = umapq_y, hue = hue, hue_order = hue_order,
                s = query_ptsize, color = query_default_color, legend = False, ax = axes, alpha = query_alpha,
                palette = query_palette, rasterized = query_rasterize
            )

        if contour_plot:
            sb.kdeplot(
                x = umapq_x, y = umapq_y, hue = chue, hue_order = contour_hue_order,
                linewidths = contour_linewidth, bw_adjust = contour_bw, bw_method = 'scott',
                fill = contour_fill, ax = axes, 
                palette = contour_palette, color = contour_default_color, alpha = contour_alpha,
                levels = contour_levels, legend = False
            )
    
        plt.xticks(fontproperties = ftprop)
        plt.yticks(fontproperties = ftprop)

        # annotations of query dataset markers.
        # under normal cases, the query data do not contain a cell type specification
        # however, we will try to plot these in case query_hue is not None

        have_legend = 'none'
        legend_order = None
        legend_colors = None
        if (query_hue is not None) and stratification == 'query' and query_legend:
            have_legend = 'query'
            legend_order = hue_order
            legend_colors = query_palette if isinstance(query_palette, list) else \
                sb.color_palette(query_palette, as_cmap = True)(np.linspace(0, 1, len(legend_order)))
            
        if (atlas_hue is not None) and stratification == 'atlas' and atlas_legend and \
            atlas_color_mode == 'categorical':

            have_legend = 'atlas'
            legend_order = ahue_order
            legend_colors = atlas_palette if isinstance(atlas_palette, list) else \
                sb.color_palette(atlas_palette, as_cmap = True)(np.linspace(0, 1, len(legend_order)))

        if have_legend in ['query', 'atlas']:
            assert len(legend_colors) == len(legend_order)
            dummy_objects = []
            legend_artists = {}
            for legend_t, legend_c, legend_id in zip(
                legend_order, legend_colors, range(len(legend_order))
            ):
                dummy = index_object()
                dummy_objects += [dummy]
                handler = index_object_handler()
                handler.set_option(legend_id + 1, legend_t, legend_c)
                legend_artists[dummy] = handler
                pass

            plt.legend(
                dummy_objects, legend_order, handler_map = legend_artists, ncol = legend_col,
                loc = 'upper left', bbox_to_anchor = (1, 1), frameon = False, prop = ftprop
            )

        # plot the color bar if expression data is used.
        if atlas_legend and (aexpr is not None):
            cmap = atlas_palette
            norm = mpl.colors.Normalize(vmin = np.min(aexpr), vmax = np.max(aexpr))

            fig.colorbar(
                cm.ScalarMappable(norm = norm, cmap = cmap),
                ax = axes, orientation = 'horizontal', shrink = 0.25
            )

        # plot the centre of all clusters on map

        do_annotate = 'none'
        annot_stype = None
        annot_stroke = None
        annot_foreground = None
        annot_x = None
        annot_y = None
        
        if stratification == 'query' and query_legend and query_annotate and legend_order is not None:
            do_annotate = 'query'
            annot_stype = query_annotate_style
            annot_stroke = query_annotate_stroke
            annot_foreground = query_annotate_foreground
            annot_x = umapq_x
            annot_y = umapq_y

        if stratification == 'atlas' and atlas_legend and atlas_annotate and legend_order is not None:
            do_annotate = 'atlas'
            annot_stype = atlas_annotate_style
            annot_stroke = atlas_annotate_stroke
            annot_foreground = atlas_annotate_foreground
            annot_x = umap_x
            annot_y = umap_y

        if do_annotate in ['query', 'atlas']:
            
            assert len(legend_colors) == len(legend_order)
            for legend_t, legend_c, legend_id in zip(
                legend_order, legend_colors, range(len(legend_order))
            ):
                # calculate gravity for legend_t class.
                selection = [x == legend_t for x in (hue if do_annotate == 'query' else ahue)]
                xs = annot_x[selection]
                ys = annot_y[selection]
                center = np.mean(xs), np.mean(ys)
                text = mtext.Text(
                    x = center[0], y = center[1], fontproperties = ftboldprop,
                    text = str(legend_id + 1) if annot_stype == 'index' else legend_t, 
                    color = annot_foreground,
                    ha = 'center', va = 'center', size = 12
                )
                
                text.set_path_effects([
                    mpe.Stroke(linewidth = 3, foreground = annot_stroke),
                    mpe.Normal()
                ])

                axes.add_artist(text)
                pass

        if elegant:
            plt.xticks(labels = [], ticks = [])
            plt.yticks(labels = [], ticks = [])
            plt.axis('off')
            show_title = atlas_gene if atlas_gene is not None else 'Embeddings'
            if title is not None:
                plt.title(
                    title if title != 'Embeddings' else show_title, 
                    fontproperties = ftboldprop
                )
    
        if save is not None:
            plt.tight_layout()
            plt.savefig(save)
            plt.close()
            return None
        
        else: 
            plt.show()
            return fig


def make_reference(
    atlas,
    var_index = None,
    key_counts = None,
    key_batch = None,
    batch_cell_filter = 5,
    taxon = 'mmu',

    # scvi model settings
    scvi_n_epoch = None,
    scvi_n_latent = 10,
    scvi_n_hidden = 128,
    scvi_n_layers = 1,
    scvi_dropout_rate = 0.1,
    scvi_dispersion = 'gene',
    scvi_gene_likelihood = 'zinb',
    scvi_latent_distrib = 'normal',

    build_umap = True,
    umap_nn = 25,
    umap_metrics = 'euclidean',
    umap_min_dist = 0.1,
    umap_init = 'spectral',

    build_parametric = False,
    parametric_encoder = default_encoder,
    save = 'reference'
):
    
    if os.path.exists(save):
        print(f'[e] the save folder already exist! specify another one.')
        return None
    else: os.makedirs(save)

    # extract count matrix.
    import anndata
    counts = anndata.AnnData(
        X = atlas.layers[key_counts] \
            if key_counts is not None else atlas.X
    )

    counts.obs['batch'] = atlas.obs[key_batch].tolist() \
        if key_batch in atlas.obs.keys() else '_whole'
    counts.var_names = atlas.var[var_index].tolist() \
        if var_index in atlas.var.keys() else atlas.var_names.tolist()

    # extract metadata file.
    import scipy.sparse as sparse
    n_cells, n_genes = counts.X.shape
    metadata = anndata.AnnData(
        X = sparse.csr_matrix((n_cells, n_genes), dtype = np.float32),
        obs = atlas.obs, var = atlas.var
    )

    metadata.var_names = atlas.var[var_index].tolist() \
        if var_index in atlas.var.keys() else atlas.var_names.tolist()
    metadata.var['.index'] = metadata.var_names.tolist()
    metadata.write_h5ad(os.path.join(save, 'metadata.h5ad'))

    # build scvi model
    import scvi
    
    # we will remove all data with < batch_cell_filter cell detection.
    mapping = {}
    names = counts.obs['batch'].value_counts().index.tolist()
    values = counts.obs['batch'].value_counts().tolist()
    n_outlier_sample = 0
    for n, v in zip(names, values):
        if v > batch_cell_filter: mapping[n] = n
        else: 
            mapping[n] = 'outliers'
            n_outlier_sample += 1
    
    print(f'[!] {n_outlier_sample} samples is removed due to small sample size.')
    batch = counts.obs['batch'].tolist()
    for i in range(len(batch)):
        batch[i] = mapping[batch[i]]
    counts.obs['batch'] = batch

    scvi.model.SCVI.setup_anndata(counts, batch_key = 'batch')
    model = scvi.model.SCVI(
        counts, 
        n_hidden = scvi_n_hidden, 
        n_latent = scvi_n_latent, 
        n_layers = scvi_n_layers,
        dropout_rate = scvi_dropout_rate,
        dispersion = scvi_dispersion,
        gene_likelihood = scvi_gene_likelihood,
        latent_distribution = scvi_latent_distrib
    )

    print('[*] scvi model config: \n')
    print(model)
    print()

    max_epochs_scvi = np.min([round((20000 / counts.n_obs) * 400), 400]) \
        if scvi_n_epoch is None else scvi_n_epoch
    print(f'[*] will train {max_epochs_scvi} epochs.')
    model.train(max_epochs = int(max_epochs_scvi), early_stopping = True)
    scvi_pc = model.get_latent_representation()
    model.save(os.path.join(save, 'scvi'))

    metadata.obsm['scvi.ref'] = scvi_pc
    metadata.uns['latent'] = {
        'batch': 'batch',
        'latent': scvi_n_latent,
        'precomputed': 'scvi.ref',
        'variable': '.index'
    }

    landmark = np.arange(n_cells, dtype = np.int64)
    np.random.shuffle(landmark)
    metadata.uns['landmark'] = landmark

    if build_umap:

        from umap import UMAP
        embedder = UMAP(
            n_components = 2, n_neighbors = umap_nn, metric = umap_metrics,
            init = umap_init, min_dist = umap_min_dist, random_state = 42
        )

        emb = embedder.fit_transform(scvi_pc)

        metadata.uns['umap'] = {
            'dimension': 2,
            'latent': scvi_n_latent,
            'precomputed': 'umap'
        }

        os.makedirs(os.path.join(save, 'umap'))
        np.save(os.path.join(save, 'umap', 'embeddings.npy'), emb)

        import pickle
        with open(os.path.join(save, 'umap', 'embedder.pkl'), 'wb') as f:
            pickle.dump(embedder, f)

    if build_parametric:

        pass

    metadata.write_h5ad(os.path.join(save, 'metadata.h5ad'))
    pass


def rebuild_umap_embedder(
    ref_path,              
    umap_nn = 25,
    umap_metrics = 'euclidean',
    umap_min_dist = 0.1,
    umap_init = 'spectral',
):
    
    from umap import UMAP
    embedder = UMAP(
        n_components = 2, n_neighbors = umap_nn, metric = umap_metrics,
        init = umap_init, min_dist = umap_min_dist, random_state = 42
    )

    import scanpy
    metadata = scanpy.read(os.path.join(ref_path, 'metadata.h5ad'))
    emb = embedder.fit_transform(metadata.obsm[metadata.uns['latent']['precomputed']])
    metadata.uns['umap'] = {
        'dimension': 2,
        'latent': metadata.uns['latent']['latent'],
        'precomputed': 'umap'
    }

    os.makedirs(os.path.join(ref_path, 'umap'), exist_ok = True)
    np.save(os.path.join(ref_path, 'umap', 'embeddings.npy'), emb)

    import pickle
    with open(os.path.join(ref_path, 'umap', 'embedder.pkl'), 'wb') as f:
        pickle.dump(embedder, f)

    metadata.write_h5ad(os.path.join(ref_path, 'metadata.h5ad'))

    pass


def rebuild_expression(ref_path, atlas, layer = 'logcounts'):

    import scanpy
    metadata = scanpy.read(os.path.join(ref_path, 'metadata.h5ad'))
    assert metadata.n_obs == atlas.n_obs
    assert metadata.n_vars == atlas.n_vars

    import anndata
    import scipy.sparse as sp
    expr = atlas.layers['layer'] if layer is not None else atlas.X
    if not isinstance(expr, sp.csc_matrix):
        expr = sp.csc_matrix(expr)
    
    data = anndata.AnnData(X = expr)
    data.write_h5ad(os.path.join(ref_path, 'expression.h5ad'))

