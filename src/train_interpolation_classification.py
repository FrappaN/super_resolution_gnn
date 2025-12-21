import argparse
import os
import torch
import pandas as pd

# convert the df into a pytorch geometric data object
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, scatter

# create a kNN graph with k=10
from torch_geometric.nn import knn_graph
from torch_geometric import seed_everything


from data_loader import *
from models import GNNPredictor
from sklearn.metrics import roc_auc_score, f1_score

import numpy as np

def create_spatial_graph(state_buss, buss_out_embs, k=10, device='cpu'):
    latitudes = torch.tensor(state_buss.latitude.values, dtype=torch.float).view(-1, 1)
    longitudes = torch.tensor(state_buss.longitude.values, dtype=torch.float).view(-1, 1)
    # convert degrees to radians
    latitudes = latitudes * (np.pi / 180)
    longitudes = longitudes * (np.pi / 180)

    # convert to x, y and z coordinates
    x = torch.cos(latitudes) * torch.cos(longitudes)
    y = torch.cos(latitudes) * torch.sin(longitudes)
    z = torch.sin(latitudes)
    pos = torch.cat([x, y, z], dim=1)


    edge_index = knn_graph(pos, k=k, loop=False)
    edge_index = to_undirected(edge_index)

    y = torch.tensor(state_buss.target.values, dtype=torch.long).view(-1)


    entity_ids = state_buss['entity_id'].values
    _, entity_ids = np.unique(entity_ids, return_inverse=True)
    entity_ids = torch.tensor(entity_ids, dtype=torch.long)

    y = scatter(y, entity_ids, dim=0, reduce='mean').long()

    data = Data(x=buss_out_embs, edge_index=edge_index, y=y, entity_ids=entity_ids).to(device)

    return data
    

def parse_args():
    parser = argparse.ArgumentParser(description="Interpolation classification over spatial graph with macroentity aggregation")
    parser.add_argument("--n_quantiles", type=int, default=5)
    parser.add_argument('--urban_areas', type=str, default='Philadelphia,St. Louis,Indianapolis,Nashville,Tampa', help='Comma-separated urban areas to include')
    parser.add_argument('--macroentities', type=str, default='bg,tract,zip', help='Comma-separated target entities (bg, tract, zip)')
    parser.add_argument('--business_csv_path', type=str, default='../datasets/yelp2019_business.csv', help='Path to businesses CSV')
    parser.add_argument('--models_dir', type=str, default='../results/yelp2019/', help='Directory containing precomputed business embeddings (*.npy)')
    parser.add_argument('--results_csv', type=str, default='../results/yelp2019/interpolation_GNN_results_classification.csv', help='Output CSV path for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=5000, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--k', type=int, default=10, help='k for kNN spatial graph')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for GNN')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--dem_vars', type=str, default='Median_Income,Median_HomeValue,Commute_Time,Median_Age', help='Comma-separated demographic variables to classify')
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)

    business_df = pd.read_csv(args.business_csv_path)

    business_df.loc[:, 'postal_code'] = business_df['postal_code'].fillna(0)
    business_df['postal_code'] = business_df['postal_code'].astype(str)
    business_df.loc[:, 'block_group_id'] = business_df['block_group_id'].fillna(0)
    business_df['block_group_id'] = business_df['block_group_id'].astype(int).astype(str)


    block_groups = business_df['block_group_id'].unique()


    # converting full names into abbreviations
    states_names_for_superres ={
        'Illinois': 'IL',
        'New Jersey': 'NJ',
        'Delaware': 'DE',
        'Missouri': 'MO',
        'Pennsylvania': 'PA',
        'Tennessee': 'TN',
        'Florida': 'FL',
        'Indiana': 'IN',
        }
    
    
    model_better_names = {"NoEdges": "Emb", "NoFeatures": "RGNN", "FeatAndEmb": "Feat-RGNN", "FeatAndEmbAndPostalCode": "Feat\&PC-RGNN", "LightGCN": "LightGCN",
                      "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN",  "NoPreTrain": "NoPreTrain"}

    
    dem_var_list = [s.strip() for s in args.dem_vars.split(',') if s.strip()]
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Commute_Time': 'Commute Time', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}
    urban_areas = [s.strip() for s in args.urban_areas.split(',') if s.strip()]

    full_bg_df = load_demographics(states_names_for_superres, block_groups)

    total_results = []
    for target_entity in [s.strip() for s in args.macroentities.split(',') if s.strip()]:
        num_trials = 10

        for urbe in urban_areas:
            for dem_var_name in dem_var_list:

                curr_buss_df = business_df.loc[business_df.urban_area == urbe].copy()
                curr_bg = curr_buss_df.block_group_id.unique()
                
                # check that all the curr_bg are in full_bg_df
                curr_bg = [bg for bg in curr_bg if bg in full_bg_df.index]
                curr_buss_df = business_df.loc[business_df.block_group_id.isin(curr_bg)].copy()
                
                # here we scale the target variable
                if target_entity == 'bg':
                    # drop any business whose block group target variable is null
                    curr_buss_df = curr_buss_df[~curr_buss_df.block_group_id.isin(full_bg_df[full_bg_df[dem_var_name].isnull()].index)]
                    curr_buss_bg = curr_buss_df.block_group_id.values
                    curr_bg = curr_buss_df.block_group_id.unique()

                    curr_values = full_bg_df.loc[curr_bg, dem_var_name].values
                    curr_unique_values = curr_values
                else:
                    # drop any business whose macroentity target variable is null
                    curr_buss_df = curr_buss_df[~curr_buss_df.block_group_id.isin(full_bg_df[full_bg_df[f'{dem_var_name}_{target_entity}'].isnull()].index)]
                    curr_buss_bg = curr_buss_df.block_group_id.values
                    curr_bg = curr_buss_df.block_group_id.unique()

                    curr_values = full_bg_df.loc[curr_bg, f'{dem_var_name}_{target_entity}'].values
                    curr_unique_values = full_bg_df.loc[curr_bg].groupby(f'{target_entity}_dcid')[f'{dem_var_name}_{target_entity}'].mean().values

                if target_entity == 'bg':
                    curr_buss_df.loc[:, 'entity_id'] = curr_buss_df.block_group_id.values
                else:
                    curr_buss_df.loc[:, 'entity_id'] = full_bg_df.loc[curr_buss_bg, f'{target_entity}_dcid'].values

                if len(curr_values) == 0:
                    curr_buss_df = curr_buss_df[['entity_id', 'longitude', 'latitude',]].copy()
                    curr_buss_df.loc[:, 'target'] = None
                    continue
                
                quantiles = np.quantile(curr_unique_values, q=np.linspace(0, 1, args.n_quantiles + 1))
                # bin the curr_values according to the quantiles
                full_bg_df.loc[curr_bg, f'binned_{dem_var_name}'] = pd.cut(curr_values, bins=quantiles, labels=False, include_lowest=True)
                assert (len(full_bg_df.loc[curr_bg, f'binned_{dem_var_name}'].unique()) == args.n_quantiles), "Error in binning the target variable"


                curr_buss_df = curr_buss_df[['entity_id', 'longitude', 'latitude', ]].copy()

                curr_buss_df.loc[:, 'target'] = full_bg_df.loc[curr_buss_bg, f'binned_{dem_var_name}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                num_datapoints = len(curr_buss_df)

                for model_name in model_better_names.keys(): #
                    print(f'Running model {model_name}, urban area {urbe}, target variable {dem_var_name}')
                    if model_name == 'NoPreTrain':
                        bus__out_embeddings = torch.arange(len(curr_buss_df))
                    else:
                        emb_path = os.path.join(args.models_dir, f'{model_name}__bus__out_embeddings.npy')
                        bus__out_embeddings = np.load(emb_path)
                        bus__out_embeddings = torch.tensor(bus__out_embeddings[curr_buss_df.index], dtype=torch.float)

                    #graph_data = create_covisit_graph(curr_buss_df, bus__out_embeddings, data, device=device)
                    graph_data = create_spatial_graph(curr_buss_df, bus__out_embeddings, k=args.k, device=device)
                    #graph_data = create_geoentity_graph(curr_buss_df, bus__out_embeddings, geoentity='tract_dcid', device=device)
                    seed_everything(args.seed)

                    curr_results = {}
                    # split businesses into train, val, test according to their entity_id
                    num_ids = len(graph_data.y)

                    for trial in range(num_trials):
                        seed_everything(trial)
                        labels = graph_data.y
                        train_idx = []
                        val_idx = []
                        test_idx = []
                        for class_id in range(args.n_quantiles):
                            class_indices = (labels == class_id).nonzero(as_tuple=True)[0]
                            # shuffle the indices
                            class_indices = class_indices[torch.randperm(len(class_indices))]
                            split_val = int(0.6 * len(class_indices))
                            splt_test = int(0.8 * len(class_indices))
                            test_idx.append(class_indices[splt_test:])
                            val_idx.append(class_indices[split_val:splt_test])
                            train_idx.append(class_indices[:split_val])
                        train_idx = torch.cat(train_idx, dim=0).to('cpu')
                        val_idx = torch.cat(val_idx, dim=0).to('cpu')
                        test_idx = torch.cat(test_idx, dim=0).to('cpu')

                        if len(train_idx) == 0 or len(test_idx) == 0 or len(val_idx) == 0:
                            if len(train_idx) == 0:
                                print(f'No train businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            if len(test_idx) == 0:
                                print(f'No test businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            if len(val_idx) == 0:
                                print(f'No val businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            continue
                        criterion = torch.nn.CrossEntropyLoss()
                        
                        if model_name == 'NoPreTrain':
                            # learn the embeddings as well
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=args.hidden_channels, num_classes=args.n_quantiles, num_nodes=len(curr_buss_df), learnable_emb=True, super_res=False).to(device)
                        else:
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=args.hidden_channels, num_classes=args.n_quantiles, super_res=False).to(device)

                        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                        best_loss = float('inf')
                        count = 0

                        model.train()
                        for epoch in (range(1, args.epochs + 1)):
                            model.train()
                            optimizer.zero_grad()
                            out = model(graph_data.x, graph_data.edge_index, bg_subgraph=graph_data.entity_ids)
                            loss = criterion(out[train_idx], graph_data.y[train_idx])
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            out = model(graph_data.x, graph_data.edge_index, bg_subgraph=graph_data.entity_ids)
                            val_loss = criterion(out[val_idx], graph_data.y[val_idx])
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_model = model.state_dict()
                                count = 0
                            else:
                                count += 1
                            if count >= args.patience:
                                break
                        model.load_state_dict(best_model)

                        model.eval()


                        pred_y = model(graph_data.x, graph_data.edge_index, bg_subgraph=graph_data.entity_ids).softmax(dim=1).cpu().detach().numpy()
                        discrete_pred_y = pred_y.argmax(axis=1)
                        train_y = graph_data.y[train_idx].cpu().detach().numpy()
                        val_y = graph_data.y[val_idx].cpu().detach().numpy()
                        test_y = graph_data.y[test_idx].cpu().detach().numpy()


                        test_f1 = f1_score(test_y, discrete_pred_y[test_idx].astype(int), average='micro')
                        val_f1 = f1_score(val_y, discrete_pred_y[val_idx].astype(int), average='micro')
                        train_f1 = f1_score(train_y, discrete_pred_y[train_idx].astype(int), average='micro')
                        

                        test_score = roc_auc_score(test_y, pred_y[test_idx], multi_class='ovr')
                        train_score = roc_auc_score(train_y, pred_y[train_idx],  multi_class='ovr')
                        val_score = roc_auc_score(val_y, pred_y[val_idx],  multi_class='ovr')


                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['target_entity'] = target_entity
                        curr_results['num_datapoints'] = num_datapoints
                        curr_results['model_embeddings'] = model_better_names[model_name]
                        curr_results['trial'] = trial

                        curr_results['train_f1'] = train_f1
                        curr_results['val_f1'] = val_f1
                        curr_results['test_f1'] = test_f1
                        
                        curr_results['train_score'] = train_score
                        curr_results['val_score'] = val_score
                        curr_results['test_score'] = test_score
                        
                        

                        total_results.append(curr_results.copy())

                    total_results_df = pd.DataFrame(total_results)

                    total_results_df.to_csv(args.results_csv)


if __name__ == "__main__":
    main()