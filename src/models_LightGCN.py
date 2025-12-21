import torch
from torch_geometric.nn.models import LightGCN
from typing import Optional, Type, Union
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import is_sparse, to_edge_index
from models import GNNRegressor

class LightGCN(LightGCN):
    def __init__(self,
        num_nodes: int,
        embedding_dim: int,
        num_layers: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha)

    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        **kwargs,
    ) -> Tensor:
        return super().get_embedding(edge_index, edge_weight)
    
    def forward(
        self,
        edge_index: Adj,
        edge_label_index: OptTensor = None,
        edge_weight: OptTensor = None,
        **kwargs,
    ) -> Tensor:
        if edge_label_index is None:
            if is_sparse(edge_index):
                edge_label_index, _ = to_edge_index(edge_index)
            else:
                edge_label_index = edge_index
        out = self.get_embedding(edge_index, edge_weight, **kwargs)

        out_src = out[edge_label_index[0]]
        out_dst = out[edge_label_index[1]]

        return (out_src * out_dst).sum(dim=-1)

class FeatLightGCN(LightGCN):
    """
    LightGCN model using as embedding also the node features.
    """
    def __init__(self,
        num_nodes: int,
        in_features: int,
        embedding_dim: int,
        num_layers: int,
        num_users: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)
        self.num_users = num_users
        self.bus_lin = torch.nn.Linear(in_features, embedding_dim)

        self.reset_parameters()
    
    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        x_bus: Tensor = None,
        **kwargs,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight

        x_bus = x[self.num_users:] + self.bus_lin(x_bus)

        # update x with the new bus features without causing grad problems
        x = torch.cat([x[:self.num_users], x_bus], dim=0)
        
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]
        return out
        
    def reset_parameters(self):
            r"""Resets all learnable parameters of the module."""
            torch.nn.init.xavier_uniform_(self.embedding.weight)
            if hasattr(self, 'bus_lin'):
                self.bus_lin.reset_parameters()
            for conv in self.convs:
                conv.reset_parameters()




class FeatAndPCLightGCN(LightGCN):
    """
    LightGCN model using as embedding also the node features.
    """
    def __init__(self,
        num_nodes: int,
        in_features: int,
        embedding_dim: int,
        num_layers: int,
        num_users: int,
        num_pcs: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)
        self.num_users = num_users
        self.bus_lin = torch.nn.Linear(in_features, embedding_dim)
        self.pc_emb = torch.nn.Embedding(num_pcs, embedding_dim)

        self.reset_parameters()
    
    def get_embedding(
        self,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        x_bus: Tensor = None,
        pcs: Tensor = None,
        **kwargs,
    ) -> Tensor:
        r"""Returns the embedding of nodes in the graph."""
        x = self.embedding.weight
        
        x_bus = x[self.num_users:] + self.bus_lin(x_bus) + self.pc_emb(pcs)
        
        x = torch.cat([x[:self.num_users], x_bus], dim=0)
        
        out = x * self.alpha[0]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_weight)
            out = out + x * self.alpha[i + 1]

        return out
        
    def reset_parameters(self):
            r"""Resets all learnable parameters of the module."""
            torch.nn.init.xavier_uniform_(self.embedding.weight)
            if hasattr(self, 'bus_lin'):
                self.bus_lin.reset_parameters()
            if hasattr(self, 'pc_emb'):
                self.pc_emb.reset_parameters()
            for conv in self.convs:
                conv.reset_parameters()


class LightGCNRegression(LightGCN):
    def __init__(self,
        encoder_class: Type[LightGCN],
        num_nodes: int,
        in_features: int,
        embedding_dim: int,
        num_layers: int,
        num_users: int,
        num_pcs: int,
        alpha: Optional[Union[float, Tensor]] = None,
        **kwargs,
    ):
        super().__init__(num_nodes, embedding_dim, num_layers, alpha, **kwargs)
        self.encoder = encoder_class(num_nodes=num_nodes, in_features=in_features, embedding_dim=embedding_dim, num_layers=num_layers, num_users=num_users, num_pcs=num_pcs, alpha=alpha, **kwargs)
        self.num_users = num_users

        self.regressor = GNNRegressor(embedding_dim, embedding_dim)
    
    def forward(
        self,
        edge_index_user_business: Adj,
        edge_index: Adj,
        edge_index_user_business_weight: OptTensor = None,
        x_bus: Tensor = None,
        pcs: Tensor = None,
        **kwargs,
    ) -> Tensor:
        embeddings = self.encoder.get_embedding(
            edge_index=edge_index_user_business,
            edge_weight=edge_index_user_business_weight,
            x_bus=x_bus,
            pcs=pcs,
            **kwargs,
        )
        bus_emb = embeddings[self.num_users:]

        out = self.regressor(
            bus_emb,
            edge_index,
        )

        return out