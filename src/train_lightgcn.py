



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
import torch.nn.functional as F
import torch_geometric.utils as tgu
from data_loader import Yelp_data_manager
from models_LightGCN import LightGCN, FeatLightGCN, FeatAndPCLightGCN
from torch_geometric.nn import MIPSKNNIndex
import tqdm
from torch_geometric.metrics import (
    LinkPredMAP,
    LinkPredPrecision,
    LinkPredRecall,
    LinkPredNDCG
)

class exp_runner:
    def __init__(self, version, sentiment=False, start_seed=0, num_trials=10, num_epochs=5, device='cuda:0', verbose=False, path='../results/'):
        self.version = version
        self.start_seed = start_seed
        self.num_trials = num_trials
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose
        self.path = path
        self.sentiment = sentiment
        return
    
    def model_init(self, data, model_name, **kwargs):

        if 'embedding_dim' in kwargs:
            embedding_dim = kwargs['embedding_dim']
        else:
            embedding_dim = 64
        if 'num_layers' in kwargs:
            num_layers = kwargs['num_layers']
        else:
            num_layers = 2

        if model_name == 'LightGCN':
            model = LightGCN(
                num_nodes = data.num_nodes,
                embedding_dim = embedding_dim,
                num_layers = num_layers,
                alpha=1/3,
            )
        elif model_name == 'FeatLightGCN':
            model = FeatLightGCN(
                num_nodes=data.num_nodes,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                in_features=data.x.size(1),
                alpha=1/3,
                num_users=self.num_users,
            )
        elif model_name == 'FeatAndPCLightGCN':
            model = FeatAndPCLightGCN(
                num_nodes=data.num_nodes,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                in_features=data.x.size(1),
                alpha=1/3,
                num_users=self.num_users,
                num_pcs=len(data.pc.unique()),
            )
        elif model_name == 'FeatAndBGLightGCN':
            model = FeatAndPCLightGCN(
                num_nodes=data.num_nodes,
                embedding_dim=embedding_dim,
                num_layers=num_layers,
                in_features=data.x.size(1),
                alpha=1/3,
                num_users=self.num_users,
                num_pcs=len(data.bg.unique()),
            )
        else:
            raise ValueError(f"Model {model_name} not recognized")

        return model

    def run_experiment(self, model_name, yelp_data, seed):

        results = {}

        seed_everything(seed)

        data = yelp_data.get_data()

        self.num_users = data.num_users
        self.num_businesses = data.num_businesses

        model = self.model_init(data, model_name=model_name)
        model.to(self.device)
        
        if self.sentiment:
            model_name = model_name+'_with_sentiment_scores'

        if 'BG' in model_name:
            data.pc = data.bg
        

        results['Model'] = model_name
        results['seed'] = seed


        yelp_data.set_data_split(seed=seed)

        ## train the model
        if self.verbose:
            print(f"Training {model_name} model")

        val_edge_label_index, train_edge_label_index, train_edge_weights = yelp_data.get_val_data()
        val_edge_label_index = val_edge_label_index.to(self.device)
        train_edge_label_index = train_edge_label_index.to(self.device)
        # train_edge_weights = train_edge_weights.to(self.device)
        if not self.sentiment:
            train_edge_weights = None

        model = self.train_model(data, model,  train_edge_label_index, val_edge_label_index, train_edge_weights)

        # test and save the results
        if self.verbose:
            print(f"Testing {model_name} model")


        test_edge_label_index, train_val_edge_label_index, train_val_edge_weights = yelp_data.get_test_data()
        exclude_links = train_val_edge_label_index.clone().to(self.device)

        train_val_edge_label_index, train_val_edge_weights = tgu.to_undirected(train_val_edge_label_index, train_val_edge_weights, num_nodes=data.num_users + data.num_businesses, reduce='mean')
    
        test_edge_label_index = test_edge_label_index.to(self.device)
        train_val_edge_label_index = train_val_edge_label_index.to(self.device)
        # train_val_edge_weights =  train_val_edge_weights.to(self.device)
        if not self.sentiment:
            train_edge_weights = None
        

        embs = model.get_embedding(train_val_edge_label_index, edge_weight=train_val_edge_weights,
                                   x_bus=data.x.to(self.device), pcs=data.pc.to(self.device))



        user_emb, bus_emb = embs[:self.num_users], embs[self.num_users:]

        metrics = self.test_model(user_emb, bus_emb, test_edge_label_index.clone(), exclude_links.clone(), k=20)
        print('MAP@20: ', metrics[0], 'Precision@20: ', metrics[1], 'Recall@20: ', metrics[2], 'NDCG@20: ', metrics[3])
        results['MAP@20'], results['Precision@20'], results['Recall@20'], results['NDCG@20'] = metrics

        # correlation between business embedding similarity and distance
        if self.verbose:
            print(f"Calculating correlation between embeddings and distances for {model_name}")
        model.eval()

        bus_emb = bus_emb.detach().cpu().numpy()

        if seed == 0:
            #saving results
            np.save(self.path+f'{model_name}__bus__out_embeddings.npy', bus_emb)
            torch.save(model.state_dict(), self.path+f'{model_name}_model.pth')

        del bus_emb, user_emb, embs, model, data

        torch.cuda.empty_cache()
    
        return results
    
    def run_multiple_experiments(self, models):

        # paper models: "LightGCN", "FeatLightGCN", "FeatAndPCLightGCN",
        for model_name in models:
            print(f"Running {model_name}")
            if self.sentiment:
                print('... with sentiment scores')
            results = []

            for seed in tqdm.trange(self.start_seed, self.start_seed+self.num_trials):
                yelp_data = Yelp_data_manager(version=self.version, seed=seed, heterogenous=False)
                results.append(self.run_experiment(model_name, yelp_data, seed))
            self.save_results(results, path=self.path)

        return

    def train_model(self, data, model, train_edge_label_index, val_edge_label_index, train_edge_weights=None):
        model = model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        counter = 0
        best_recall = 0.

        train_data = data.clone()
        exclude_links = train_edge_label_index.clone()
        train_data.edge_index, train_data.edge_weight =  tgu.to_undirected(train_edge_label_index, train_edge_weights, num_nodes=data.num_users + data.num_businesses, reduce='mean')

        train_data.to(self.device)

        for epoch in tqdm.tqdm(range(self.num_epochs)):
            total_loss = total_examples = 0
            model.train()

            pos_edge_label_index = train_edge_label_index.clone()

            neg_edge_label_index = torch.stack([
                pos_edge_label_index[0],
                torch.randint(self.num_users, self.num_users + self.num_businesses,
                            (pos_edge_label_index[0].numel(), ), device=self.device)
            ], dim=0)

            edge_label_index = torch.cat([
                pos_edge_label_index,
                neg_edge_label_index,
            ], dim=1).to(self.device)
            optimizer.zero_grad()
            pos_rank, neg_rank = model(edge_index=train_data.edge_index, edge_label_index=edge_label_index, edge_weight=train_data.edge_weight, x_bus=train_data.x, pcs=train_data.pc).chunk(2)
            loss = model.recommendation_loss(
                pos_rank,
                neg_rank,
                node_id=edge_label_index.unique(),
            )
            loss.backward()
            optimizer.step()

            if epoch >= 1000 and epoch % 100 == 0:
                model.eval()
                
                embs = model.get_embedding(train_data.edge_index, edge_weight=train_data.edge_weight, x_bus=train_data.x, pcs=train_data.pc)

                user_emb, bus_emb = embs[:self.num_users], embs[self.num_users:]
                val_metrics = self.test_model(user_emb, bus_emb, val_edge_label_index.clone(), exclude_links.clone(), k=20)
                val_map, val_prec, val_recall, val_ndcg = val_metrics
                print(f'Epoch: {epoch}, Loss: {loss}, Recall@20: {val_recall}, MAP@20: {val_map}, Precision@20: {val_prec}, NDCG@20: {val_ndcg}')
                if val_recall > best_recall:
                    best_recall = val_recall
                    best_model = model.state_dict()
                    counter = 0
                else:
                    counter += 100
                if counter >= 500:
                    break
        del exclude_links
        #torch.cuda.empty_cache()
        
        model.load_state_dict(best_model)
        return model



    def test_model(self, src_emb, dst_emb, edge_label_index, exclude_links, k=20):

        # Initialize metrics:
        #print('initializing metrics')
        map_metric = LinkPredMAP(k=k).to(self.device)
        precision_metric = LinkPredPrecision(k=k).to(self.device)
        recall_metric = LinkPredRecall(k=k).to(self.device)
        ndcg_metric = LinkPredNDCG(k=k).to(self.device)

        batch_size = 1024
        num_users = src_emb.size(0)
        
        exclude_links[1] = exclude_links[1] - self.num_users
        edge_label_index[1] = edge_label_index[1] - self.num_users

        for start in range(0, num_users, batch_size):
            end = start + batch_size
            emb = src_emb[start:end]
            
            logits_matrix = torch.matmul(emb, dst_emb.t())

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
            _, pred_index = logits_matrix.topk(k, dim=1)

            map_metric.update(pred_index, _edge_label_index)
            precision_metric.update(pred_index, _edge_label_index)
            recall_metric.update(pred_index, _edge_label_index)
            ndcg_metric.update(pred_index, _edge_label_index)
        
        del logits_matrix, edge_label_index, exclude_links
        #torch.cuda.empty_cache()

        return (
            float(map_metric.compute()),
            float(precision_metric.compute()),
            float(recall_metric.compute()),
            float(ndcg_metric.compute())
        )
    
    def save_results(self, results, path='results/'):
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
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--verbose', type=bool, default=False)
    parser.add_argument('--version', '-v',  type=str, default='yelp2019')
    parser.add_argument('--sentiment', default=False, action='store_true') 
    parser.add_argument('--results_path', type=str, default='../results/')
    parser.add_argument('--models', '-m', nargs='+', default=[])


    args = parser.parse_args()

    if torch.cuda.is_available():

        device = args.device
        device = 'cuda'

    start_seed = args.start_seed
    num_trials = args.num_trials
    num_epochs = args.num_epochs
    version = args.version
    sentiment = args.sentiment
    models = args.models

    verbose = args.verbose
    path = args.results_path + version + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    # run experiments 
    print('Running experiments')
    runner = exp_runner(version=version, sentiment=sentiment, start_seed=start_seed, num_trials=num_trials, num_epochs=num_epochs, device=device, verbose=verbose,  path=path)
    runner.run_multiple_experiments(models)


if __name__ == '__main__':
    main()