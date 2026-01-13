from torch_geometric.nn import SAGEConv, GATConv, to_hetero, LGConv, GCNConv
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv
from torch_geometric.nn.models.lightgcn import BPRLoss
import torch
from torch import Tensor
from torch_geometric.nn.models import MLP
import torch_geometric.nn as tgnn
from typing import Optional

class Classifier(torch.nn.Module):
    def forward(self, x_user: Tensor, x_business: Tensor, edge_label_index: Tensor = None) -> Tensor:

        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_business = x_business[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        output = (edge_feat_user * edge_feat_business).sum(dim=-1)
        return output
    
class GeneralClassifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.affine_output = torch.nn.Linear(in_features=hidden_channels, out_features=1)

    def forward(self, x_user: Tensor, x_business: Tensor, edge_label_index: Tensor = None) -> Tensor:

        # Convert node embeddings to edge-level representations:
        edge_feat_user = x_user[edge_label_index[0]]
        edge_feat_business = x_business[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        output = self.affine_output(torch.mul(edge_feat_user, edge_feat_business))
        return output.reshape(-1)


#define an abstract class for my models
class AbstractRepModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = None
        self.decoder = Classifier()
        return

    def forward(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_dict = self.encoder(x_dict, data.edge_index_dict)

        pred = self.decoder(
            x_dict["user"],
            x_dict["business"],
            data["user", "rates", "business"].edge_label_index,
        )
        return pred
    
    def forward_test(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_dict = self.encoder(x_dict, data.edge_index_dict)

        pred = self.decoder(
            x_dict["user"],
            x_dict["business"],
        )
        return pred
    
    def get_embeddings(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_dict = self.encoder(x_dict, data.edge_index_dict)
        
        return x_dict

    def recommendation_loss(
        self,
        pos_edge_rank: Tensor,
        neg_edge_rank: Tensor,
        user_node_id: Optional[Tensor] = None,
        bus_node_id: Optional[Tensor] = None,
        lambda_reg: float = 1e-4,
        **kwargs,
    ) -> Tensor:
        r"""Computes the model loss for a ranking objective via the Bayesian
        Personalized Ranking (BPR) loss.

        .. note::

            The i-th entry in the :obj:`pos_edge_rank` vector and i-th entry
            in the :obj:`neg_edge_rank` entry must correspond to ranks of
            positive and negative edges of the same entity (*e.g.*, user).

        Args:
            pos_edge_rank (torch.Tensor): Positive edge rankings.
            neg_edge_rank (torch.Tensor): Negative edge rankings.
            node_id (torch.Tensor): The indices of the nodes involved for
                deriving a prediction for both positive and negative edges.
                If set to :obj:`None`, all nodes will be used.
            lambda_reg (int, optional): The :math:`L_2` regularization strength
                of the Bayesian Personalized Ranking (BPR) loss.
                (default: :obj:`1e-4`)
            **kwargs (optional): Additional arguments of the underlying
                :class:`torch_geometric.nn.models.lightgcn.BPRLoss` loss
                function.
        """
        loss_fn = BPRLoss(lambda_reg, **kwargs)
        user_emb = self.user_emb.weight
        user_emb = user_emb if user_node_id is None else user_emb[user_node_id]
        if hasattr(self, 'bus_emb'):
            bus_emb = self.bus_emb.weight
            bus_emb = bus_emb if bus_node_id is None else bus_emb[bus_node_id]
            emb = torch.cat([user_emb, bus_emb], dim=0)
        else:
            emb = user_emb
        return loss_fn(pos_edge_rank, neg_edge_rank, emb)
    
    def feat_encoder(self, data: HeteroData) ->  "tuple[dict, dict]" :
        raise NotImplementedError("Subclasses should implement this!")
    

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    
class GNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, num_nodes=None, learnable_emb=False, super_res=True):
        super().__init__()
        if learnable_emb:
            self.node_emb = torch.nn.Embedding(num_nodes, in_channels)
        self.super_res = super_res
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.lin_out = torch.nn.Linear(hidden_channels, hidden_channels)
        self.decoder = torch.nn.Linear(hidden_channels, num_classes)

    def forward(self, x: Tensor, edge_index: Tensor, subgraphs: Tensor = None, **kwargs) -> Tensor:
        # Classification after first pooling layer only
        if hasattr(self, 'node_emb'):
            x = self.node_emb(x)
        x = (self.conv1(x, edge_index)).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.decoder(x)
        
        if subgraphs is not None:
            # First pooling using bg_subgraph
            x = tgnn.global_mean_pool(x, subgraphs)
            #x = self.lin_out(x).relu()

        return x


class GAT(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = GATConv(hidden_channels, hidden_channels//2, heads=2, add_self_loops=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels//2, heads=2, add_self_loops=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None,  **kwargs) -> Tensor:
        # Define a 2-layer GNN computation graph.
        # Use a *single* `ReLU` non-linearity in-between.
        x = self.conv1(x, edge_index, edge_attr).relu()
        x = self.conv2(x, edge_index, edge_attr)
        return x
    
class MLPModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, metadata):
        super().__init__()

        # Embeddings for both users and business
        self.bus_emb = torch.nn.Embedding(num_business, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # instantiate MLP
        self.decoder = GeneralClassifier(hidden_channels)
        #self.encoder = MLP(channel_list = [hidden_channels, hidden_channels, hidden_channels])
        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_emb(data["business"].node_id),
        }
        return x_dict
    
    def forward(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_user = x_dict['user'] #self.encoder(x_dict['user'])
        x_business = x_dict['business'] #self.encoder(x_dict['business'])

        pred = self.decoder(
            x_user,
            x_business,
            data["user", "rates", "business"].edge_label_index,
        )
        return pred
    
    def forward_test(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_user = x_dict['user'] #self.encoder(x_dict['user'])
        x_business = x_dict['business'] #self.encoder(x_dict['business'])

        pred = self.decoder(
            x_user,
            x_business,
        )
        return pred
    
    def get_embeddings(self, data: HeteroData) -> Tensor:

        x_dict = self.feat_encoder(data)

        x_user = x_dict['user'] #self.encoder(x_dict['user'])
        x_business = x_dict['business'] #self.encoder(x_dict['business'])


        x_dict = {'user': x_user, 'business': x_business}
        return x_dict


class NoFeatSageModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, metadata):
        super().__init__()

        # Embeddings for both users and business
        self.bus_emb = torch.nn.Embedding(num_business, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_emb(data["business"].node_id),
        }
        return x_dict



class SageModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, metadata):
        super().__init__()

        # Linear transform on business features, embeddings for users
        self.bus_lin = torch.nn.Linear(in_features, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_lin(data["business"].x),
        }
        return x_dict
    
class GatModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, metadata):
        super().__init__()

        self.bus_lin = torch.nn.Linear(in_features, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GAT(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = self.feat_encoder(data)

        x_dict = self.encoder(x_dict, data.edge_index_dict,  data.edge_attr_dict)

        pred = self.decoder(
            x_dict["user"],
            x_dict["business"],
            data["user", "rates", "business"].edge_label_index,
        )
        return pred
    
    def forward_test(self, data: HeteroData) -> Tensor:
        x_dict = self.feat_encoder(data)

        x_dict = self.encoder(x_dict, data.edge_index_dict,  data.edge_attr_dict)

        pred = self.decoder(
            x_dict["user"],
            x_dict["business"],
        )
        return pred
    

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_lin(data["business"].x),
        }
        return x_dict


    

class PostalCodeSumModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_pcs, metadata, mode='normal'):
        super().__init__()


        self.bus_lin = torch.nn.Linear(in_features, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)
    
        self.bus_emb = torch.nn.Embedding(num_pcs, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_lin(data["business"].x)+self.bus_emb(data["business"].pc),
        }
        return x_dict
    
class PostalCodeConcatModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_pcs, metadata, mode='normal'):
        super().__init__()

        self.bus_lin = torch.nn.Linear(in_features, hidden_channels//2)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        self.bus_emb = torch.nn.Embedding(num_pcs, hidden_channels//2)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": torch.cat([self.bus_lin(data["business"].x),self.bus_emb(data["business"].pc)], dim=1),
        }

        return x_dict


class NoFeatPostalCodeConcatModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, num_pcs, metadata, mode='normal'):
        super().__init__()

        self.bus_lin = torch.nn.Linear(in_features, hidden_channels//2)
        self.bus_emb = torch.nn.Embedding(num_business, hidden_channels//2)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        self.pc_emb = torch.nn.Embedding(num_pcs, hidden_channels//2)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": torch.cat([self.bus_lin(data["business"].x),self.pc_emb(data["business"].pc)], dim=1),
        }

        return x_dict


class FeatAndEmbModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, metadata):
        super().__init__()

        # Linear transform on business features, embeddings for users
        self.bus_lin = torch.nn.Linear(in_features, hidden_channels)
        self.bus_emb = torch.nn.Embedding(num_business, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_lin(data["business"].x)+self.bus_emb(data["business"].node_id),
        }
        return x_dict
    

    
class FeatAndEmbAndPostalCodeModel(AbstractRepModel):
    def __init__(self, in_features, hidden_channels, num_users, num_business, num_pcs, metadata):
        super().__init__()

        # Linear transform on business features, embeddings for users
        self.bus_lin = torch.nn.Linear(in_features, hidden_channels)
        self.bus_emb = torch.nn.Embedding(num_business, hidden_channels)
        self.pc_emb = torch.nn.Embedding(num_pcs, hidden_channels)
        self.user_emb = torch.nn.Embedding(num_users, hidden_channels)

        # Instantiate homogeneous GNN:
        gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.encoder = to_hetero(gnn, metadata=metadata)

        return

    def feat_encoder(self, data: HeteroData) -> Tensor:
        x_dict = {
          "user": self.user_emb(data["user"].node_id),
          "business": self.bus_lin(data["business"].x)+self.bus_emb(data["business"].node_id)+self.pc_emb(data["business"].pc),
        }
        return x_dict