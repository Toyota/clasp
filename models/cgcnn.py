import sys
import torch
import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.nn import global_max_pool, global_mean_pool

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea,
             nbr_fea
             ],dim=2
        )
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out

class CGCNN(torch.nn.Module):
    """
    Model to encode crystal structures with crystal graph convolutional networks(CGCNN).

    Args:
    - configs: argparse.Namespace
        Namespace containing hyperparameters.
    - output_intermediate_feat: bool(False)
        Whether to return intermediate features or not.
    """
    
    def __init__(self, configs, output_intermediate_feat=False):
        super(CGCNN, self).__init__()

        # Retrieve parameters from `params`, falling back to defaults if not provided.
        self.params = configs
        orig_atom_fea_len = getattr(configs, 'orig_atom_fea_len', 92)
        nbr_fea_len = getattr(configs, 'nbr_fea_len', 41)
        atom_fea_len = getattr(configs, 'atom_fea_len', 64)
        embedding_dim = getattr(configs, 'embedding_dim', None)  # Assuming this must be provided by params
        n_conv_cgcnn = getattr(configs, 'n_conv_cgcnn', 3)
        crystalencoder_n_mlp_layers = getattr(configs, 'crystalencoder_n_mlp_layers', 2)
        dropout_prob = getattr(configs, 'crystalencoder_dropout_prob', 0.0)
        self.output_intermediate_feat = output_intermediate_feat

        # Crystal structure embeddings
        self.atom_embedding_proj = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convolution_layers = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                                           nbr_fea_len=nbr_fea_len)
                                                 for _ in range(n_conv_cgcnn)])
        self.crystal_embedding_proj = nn.Linear(atom_fea_len, embedding_dim)
        self.crystal_embedding_bn = nn.BatchNorm1d(num_features=embedding_dim)
        
        self.intermediate_modules = nn.ModuleList()
        for i in range(crystalencoder_n_mlp_layers):
            self.intermediate_modules.extend([
                nn.Linear(embedding_dim, embedding_dim),
                nn.ReLU(inplace=True),
                nn.BatchNorm1d(num_features=embedding_dim)
            ])
            if dropout_prob > 0:
                self.intermediate_modules.append(nn.Dropout(p=dropout_prob))

        if not hasattr(configs, 'targets') or configs.targets is None:
            # Embedding mode when `params.targets=None`.
            self.final_projection = nn.Linear(embedding_dim, embedding_dim)
        else:
            # Regression mode when targets = "hoge" or ["hoge", "foo"]
            final_dim = 1 if isinstance(configs.targets, str) \
                else len(configs.targets)
            self.regression_layer = nn.Linear(embedding_dim, final_dim)


    def forward(self, data):
        """
        Performs forward pass.

        Args:
        - data: torch_geometric.Data
            Data object containing crystal graph information.

        Returns:
        - out: torch.Tensor
            Encoded tensor with shape 'embedding_dim' after pooling of crystal.
        """

        # Convert torch_geometric.Data to CGCNN format
        N, E = data.x.shape[0], data.edge_attr.shape[0]
        atom_fea = data.x
        nbr_fea = data.edge_attr.reshape(N, E//N, -1)
        nbr_fea_idx = data.edge_index[1].reshape(N, E//N) # Use only col indices

        atom_fea = self.atom_embedding_proj(atom_fea)
        for conv_layer in self.convolution_layers:
            atom_fea = conv_layer(atom_fea, nbr_fea, nbr_fea_idx)
        
        x_crystal = global_mean_pool(atom_fea, data.batch)
        del atom_fea

        x_crystal = self.crystal_embedding_proj(x_crystal)
        x_crystal = F.relu(self.crystal_embedding_bn(x_crystal))

        # for encoder
        intermediate_features = []
        for module in self.intermediate_modules:
            x_crystal = module(x_crystal)
            if self.output_intermediate_feat and isinstance(module, nn.BatchNorm1d):
                intermediate_features.append(x_crystal)

        if self.output_intermediate_feat:
            return tuple(intermediate_features)

        if hasattr(self, 'regression_layer'):
            return self.regression_layer(x_crystal)
        
        x_crystal = self.final_projection(x_crystal)

        return x_crystal
