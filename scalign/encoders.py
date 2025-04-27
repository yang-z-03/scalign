
import torch.nn as nn
import numpy as np
from scalign.parametric import default_encoder


class resnet_encoder(nn.Module):
    
    def __init__(self, dims, n_nodes = 2048, n_layers = 4, n_components = 2):
        super().__init__()

        self.layers = ['flatten', 'preproc', 'relu_0']
        self.flatten = nn.Flatten().cuda()
        self.preproc = nn.Linear(np.prod(dims), n_nodes).cuda()
        self.relu_0 = nn.ReLU()

        self.n_layers = n_layers
        for i in range(1, n_layers + 1):
            setattr(self, 'layer_first_' + str(i), nn.Linear(n_nodes, n_nodes).cuda())
            setattr(self, 'relu_mid_' + str(i), nn.ReLU().cuda())
            setattr(self, 'layer_second_' + str(i), nn.Linear(n_nodes, n_nodes).cuda())
            setattr(self, 'relu_out_' + str(i), nn.ReLU().cuda())
            self.layers += [
                'layer_first_' + str(i), 'relu_mid_' + str(i),
                'layer_second_' + str(i), 'relu_out_' + str(i)
            ]
        
        self.finalize = nn.Linear(n_nodes, n_components).cuda()
        self.layers += ['finalize']
    
    def forward(self, X):
        z = self.flatten(X)
        z = self.preproc(z)
        z = self.relu_0(z)

        for i in range(1, self.n_layers + 1):
            weights1 = getattr(self, f'layer_first_{i}')
            relu1 = getattr(self, f'relu_mid_{i}')
            weights2 = getattr(self, f'layer_second_{i}')
            relu2 = getattr(self, f'relu_out_{i}')
            z = relu2(z + weights2(relu1(weights1(z))))
        
        return self.finalize(z)


predefined_encoders = [
    default_encoder,
    resnet_encoder
]
