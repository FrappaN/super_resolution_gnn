


import argparse
import os

# limit cpu usage

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"




import numpy as np
import pandas as pd
import scipy.spatial.distance as dist
import torch
from torch_geometric import seed_everything
from torch_geometric.nn import MIPSKNNIndex
import torch.nn.functional as F
from data_loader import Yelp_data_manager
from models import NoFeatSageModel, SageModel, PostalCodeConcatModel, MLPModel, NoFeatPostalCodeConcatModel, FeatAndEmbModel, FeatAndEmbAndPostalCodeModel, GatModel
from sklearn.metrics import roc_auc_score, r2_score

import tqdm
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
    LinkPredNDCG,
)

class exp_runner:
    def __init__(self,version, start_seed=0, num_trials=10, num_epochs=5, device='cuda:0', verbose=False, path='../results/'):
        self.version = version
        self.start_seed = start_seed
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose
        self.path = path
        return
    
    def model_init(self, data, model_name):

        if model_name == "NoFeatures":
            model = NoFeatSageModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                metadata=data.metadata()
                )
        elif model_name == "FeaturesOnly":
            model = SageModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                metadata=data.metadata()
                )
        elif model_name == "FeaturesAndPostalCode":
            model = PostalCodeConcatModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_pcs = len(data["business"].pc.unique()), 
                metadata=data.metadata()
                )
        elif model_name == "NoEdges":
            model = MLPModel(
                in_features=data['business'].x.shape[1],
                hidden_channels=64,
                num_users= data["user"].num_nodes,
                num_business= data["business"].num_nodes,
                metadata=data.metadata()
            )
        elif model_name == "NoFeatAndPostalCode":
            model = NoFeatPostalCodeConcatModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes,
                num_pcs = len(data["business"].pc.unique()), 
                metadata=data.metadata()
                )
        elif model_name == "FeatAndEmb":
            model = FeatAndEmbModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                metadata=data.metadata()
                )
        elif model_name == "FeatAndEmbAndPostalCode":
            model = FeatAndEmbAndPostalCodeModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                num_pcs = len(data["business"].pc.unique()), 
                metadata=data.metadata()
                )
        elif model_name == "FeatAndEmbAndBG":
            model = FeatAndEmbAndPostalCodeModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                num_pcs = len(data["business"].bg.unique()), 
                metadata=data.metadata()
                )
        elif model_name == "GAT":
            model = GatModel(
                in_features=data['business'].x.shape[1], 
                hidden_channels=64, 
                num_users= data["user"].num_nodes, 
                num_business= data["business"].num_nodes, 
                metadata=data.metadata()
                )
        else:
            raise ValueError(f"Model {model_name} not recognized")

        return model



    def run_experiment(self, model_name, yelp_data, seed):

        results = {}
        results['Model'] = model_name
        results['seed'] = seed

        seed_everything(seed)

        data = yelp_data.get_data()
        if 'BG' in model_name:
            data["business"].pc = data["business"].bg

        train_data, val_data, test_data = yelp_data.split_and_loaders(seed)

        ## train the model
        if self.verbose:
            print(f"Training {model_name} model")

        model = self.model_init(data, model_name)

        model = self.train_model(train_data, val_data, model)

        # test and save the results
        if self.verbose:
            print(f"Testing {model_name} model")


        edge_label_index, exclude_links, _ = yelp_data.get_test_data()
        edge_label_index = edge_label_index.to(self.device)
        exclude_links = exclude_links.to(self.device)

        # embs = model.get_embeddings(data.to(self.device))
        model.eval()

        device = self.device

        embs = model.get_embeddings(test_data.to(device))
        src_emb, dst_emb = embs["user"], embs["business"]

        metrics = self.test_model(src_emb, dst_emb, edge_label_index, exclude_links)

        results['MAP@20'], results['Precision@20'], results['Recall@20'], results['NDCG@20'] = metrics
        print(f"MAP@20: {results['MAP@20']}, Precision@20: {results['Precision@20']}, Recall@20: {results['Recall@20']}, NDCG@20: {results['NDCG@20']}")

        # correlation between business embedding similarity and distance
        if self.verbose:
            print(f"Calculating correlation between embeddings and distances for {model_name}")

        bus__out_embeddings = dst_emb.detach().cpu().numpy()

        if seed == 0 and self.path is not None:
            #saving results
            np.save(self.path+f'{model_name}__bus__out_embeddings.npy', bus__out_embeddings)
            torch.save(model.state_dict(), self.path+f'{model_name}_model.pth')

        del data, model
        
        return results
    
    def run_multiple_experiments(self, models):
        # paper models:  "NoFeatures", "FeatAndEmb","FeatAndEmbAndPostalCode", "NoEdges",
        for model_name in models:  
            print(f"Running {model_name}")
            results = []

            for seed in tqdm.trange(self.start_seed, self.start_seed+self.num_trials):
                if model_name == 'GAT':
                    yelp_data = Yelp_data_manager(version=self.version, seed=seed, heterogenous=True, with_attr=True)
                else:
                    yelp_data = Yelp_data_manager(version=self.version, seed=seed, heterogenous=True, with_attr=False)

                results.append(self.run_experiment(model_name, yelp_data, seed))
                del yelp_data
                torch.cuda.empty_cache()
            self.save_results(results, path=self.path)


        return

    def train_model(self, train_data, val_data, model):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        counter = 0
        best_loss = np.inf
        best_recall = 0

        pos_edge_label_index = train_data["user", "rates", "business"].edge_label_index.clone()


        for epoch in (pbar := tqdm.tqdm(range(self.num_epochs))):
            total_loss = total_examples = 0
            model.train()
            #for index in train_loader:
            optimizer.zero_grad()

                # modify train_data.edge_label_index to add random negative samples
            #cur_pos_edge_label_index = pos_edge_label_index[:, index]

            neg_edge_label_index = torch.zeros_like(pos_edge_label_index)
            neg_edge_label_index[0] = pos_edge_label_index[0] #torch.randint(0, train_data["user"].num_nodes, (len(pos_edge_label_index[0]),))
            neg_edge_label_index[1] = torch.randint(0, train_data["business"].num_nodes, (len(pos_edge_label_index[0]),))
            

            train_data["user", "rates", "business"].edge_label_index = torch.cat([pos_edge_label_index, neg_edge_label_index], dim=1)
            train_data["user", "rates", "business"].edge_label = torch.cat([torch.ones_like(pos_edge_label_index[0]), torch.zeros_like(pos_edge_label_index[0])])

            pred = model(train_data.to(self.device))
            pos_edge_rank, neg_edge_rank = pred.chunk(2)

            #loss = F.binary_cross_entropy_with_logits(pred, train_data["user", "rates", "business"].edge_label)
            loss = model.recommendation_loss(
                    pos_edge_rank, 
                    neg_edge_rank,
                    user_node_id=train_data["user", "rates", "business"].edge_index[0].unique(),
                    bus_node_id=train_data["user", "rates", "business"].edge_index[1].unique()
                    )

            loss.backward()
            optimizer.step()

            
            if epoch >= 1000 and epoch % 100 == 0:
                model.eval()
                embs = model.get_embeddings(val_data.to(self.device))
                src_emb, dst_emb = embs["user"], embs["business"]
                val_metrics = self.test_model(src_emb, dst_emb, val_data["user", "rates", "business"].edge_label_index, val_data["user", "rates", "business"].edge_index)
                
                val_recall = val_metrics[2]
                if val_recall > best_recall:
                    best_recall = val_recall
                    best_model = model.state_dict()
                    counter = 0
                else:
                    counter += 100
                if counter >= 500:
                    break
            # if epoch%10 == 0:
            #     print('Train loss:', loss.item(), 'Val loss:', val_loss.item())
                #pbar.set_description(f"Best loss: {best_loss:.4f}, Val loss: {val_loss:.4f}")
                pbar.set_description(f"Best recall: {best_recall:.4f}, Val recall: {val_recall:.4f}")
            

        model.load_state_dict(best_model)
        del train_data, val_data
        return model



    def test_model(self, src_emb, dst_emb, edge_label_index, exclude_links, k=20):

        # metric computators

        map_metric = LinkPredMAP(k=k).to(self.device)
        precision_metric = LinkPredPrecision(k=k).to(self.device)
        recall_metric = LinkPredRecall(k=k).to(self.device)
        ndcg_metric = LinkPredNDCG(k=k).to(self.device)

        batch_size = 1024
        num_users = src_emb.size(0)
        
        for start in range(0, num_users, batch_size):
            end = start + batch_size
            emb = src_emb[start:end]
            
            logits_matrix = torch.matmul(emb, dst_emb.t())

            # Filter labels/exclusion by current batch:
            _edge_label_index = edge_label_index.sparse_narrow(
                dim=0,
                start=start,
                length=emb.size(0),
            )
            _exclude_links = exclude_links.sparse_narrow(
                dim=0,
                start=start,
                length=emb.size(0),
            )

            logits_matrix[_exclude_links[0], _exclude_links[1]] = -float('inf')
            #print(logits_matrix.size())
            _, pred_index_mat = logits_matrix.topk(k, dim=1)

            # Update retrieval metrics:
            map_metric.update(pred_index_mat, _edge_label_index)
            precision_metric.update(pred_index_mat, _edge_label_index)
            recall_metric.update(pred_index_mat, _edge_label_index)
            ndcg_metric.update(pred_index_mat, _edge_label_index)
            del emb, pred_index_mat
            
        return (
            float(map_metric.compute()),
            float(precision_metric.compute()),
            float(recall_metric.compute()),
            float(ndcg_metric.compute()),
        )
    
    def save_results(self, results, path='results/'):
        if path is not None:
            results_df = pd.DataFrame(results)
            model_name = results[0]['Model']
            if not os.path.exists(path):
                os.makedirs(path)
            results_df.to_csv(f'{path}{model_name}_results.csv', index=False)
        return


def main():

    parser = argparse.ArgumentParser(description='Embedding Correlations')
    parser.add_argument('--start_seed', type=int, default=0)
    parser.add_argument('--num_trials', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=5000)
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--version', '-v',  type=str, default='yelp2019')
    parser.add_argument('--results_path', type=str, default='../results/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--models', '-m', nargs='+', default=[])

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    start_seed = args.start_seed
    num_trials = args.num_trials
    num_epochs = args.num_epochs
    version = args.version

    verbose = args.verbose
    debug = args.debug
    if not debug:
        path = args.results_path + version + '/'
        if not os.path.exists(path):
            os.makedirs(path)
    else:
        path = None
        num_trials=1
        num_epochs=0

    # run experiments
    print('Running experiments')
    runner = exp_runner(version=version, start_seed=start_seed, num_trials=num_trials, num_epochs=num_epochs, device=device, verbose=verbose,  path=path)
    runner.run_multiple_experiments(models=args.models)


if __name__ == '__main__':
    main()