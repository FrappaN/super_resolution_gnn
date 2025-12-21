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


from sklearn.preprocessing import QuantileTransformer
from data_loader import *
from models import GNNPredictor
from sklearn.metrics import r2_score, mean_squared_error

import numpy as np

def create_spatial_graph(state_buss, buss__out_embs, k=10, device='cpu'):
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

    y_raw = torch.tensor(state_buss.target.values, dtype=torch.float).view(-1)

    entity_ids_np = state_buss['entity_id'].values
    _, entity_ids_np = np.unique(entity_ids_np, return_inverse=True)
    entity_ids = torch.tensor(entity_ids_np, dtype=torch.long)

    # aggregate labels at entity level (mean of business targets)
    y_ent = scatter(y_raw, entity_ids, dim=0, reduce='mean').view(-1, 1)

    data = Data(x=buss__out_embs, edge_index=edge_index, y=y_ent, entity_ids=entity_ids).to(device)

    return data

def parse_args():
    parser = argparse.ArgumentParser(description="Interpolation regression over spatial graph with macroentity aggregation")
    parser.add_argument('--urban_areas', type=str, default='Philadelphia,St. Louis,Indianapolis,Nashville,Tampa', help='Comma-separated urban areas to include')
    parser.add_argument('--macroentities', type=str, default='bg,tract,zip', help='Comma-separated target entities (bg, tract, zip)')
    parser.add_argument('--business_csv_path', type=str, default='../datasets/yelp2019_business.csv', help='Path to businesses CSV')
    parser.add_argument('--models_dir', type=str, default='../results/yelp2019/', help='Directory containing precomputed business embeddings (*.npy)')
    parser.add_argument('--results_csv', type=str, default='../results/yelp2019/interpolation_GNN_results.csv', help='Output CSV path for results')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--epochs', type=int, default=5000, help='Max training epochs')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')
    parser.add_argument('--k', type=int, default=10, help='k for kNN spatial graph')
    parser.add_argument('--hidden_channels', type=int, default=32, help='Hidden channels for GNN')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay')
    parser.add_argument('--dem_vars', type=str, default='Median_Income,Median_HomeValue,Commute_Time,Median_Age', help='Comma-separated demographic variables to regress')
    parser.add_argument('--n_quantiles', type=int, default=10, help='Number of quantiles for target scaling with QuantileTransformer')
    parser.add_argument('--trials', type=int, default=10, help='Number of trials per configuration')
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
                      "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN",  "NoPretrain": "NoPretrain"}

    
    dem_var_list = [s.strip() for s in args.dem_vars.split(',') if s.strip()]
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Commute_Time': 'Commute Time', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}
    urban_areas = [s.strip() for s in args.urban_areas.split(',') if s.strip()]

    full_bg_df = load_demographics(states_names_for_superres, block_groups)

    total_results = []
    for target_entity in [s.strip() for s in args.macroentities.split(',') if s.strip()]:
        num_trials = args.trials

        urbe = None
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

                ##quantiles = min((len(curr_values), 1000))
                quantiles = args.n_quantiles
                target_scaler = QuantileTransformer(n_quantiles=quantiles, output_distribution='normal')
                target_scaler.fit(curr_unique_values.reshape(-1, 1))
                full_bg_df.loc[curr_bg, f'scaled_{dem_var_name}'] = target_scaler.transform(curr_values.reshape(-1, 1))


                curr_buss_df = curr_buss_df[['entity_id', 'longitude', 'latitude', ]].copy()

                curr_buss_df.loc[:, 'target'] = full_bg_df.loc[curr_buss_bg, f'scaled_{dem_var_name}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                num_datapoints = len(curr_buss_df)

                for model_name in model_better_names.keys(): #
                    print(f'Running model {model_name}, urban area {urbe}, target variable {dem_var_name}')
                    if model_name == 'NoPretrain':
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
                    # split by entity-level indices (60/20/20)
                    num_ids = graph_data.y.shape[0]
                    perm = torch.randperm(num_ids)
                    split_train = int(0.6 * num_ids)
                    split_val = int(0.8 * num_ids)
                    train_idx = perm[:split_train]
                    val_idx = perm[split_train:split_val]
                    test_idx = perm[split_val:]

                    if len(train_idx) == 0 or len(test_idx) == 0 or len(val_idx) == 0:
                        print(f'With {num_ids} unique entities')
                        if len(train_idx) == 0:
                            print(f'No train entities in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                        if len(test_idx) == 0:
                            print(f'No test entities in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                        if len(val_idx) == 0:
                            print(f'No val entities in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                        continue

                    for trial in range(num_trials):

                        # learn the model on the urban area
                        criterion = torch.nn.MSELoss()
                        if model_name == 'NoPretrain':
                            # learn the embeddings as well
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=32, num_classes=1, num_nodes=len(curr_buss_df), learnable_emb=True, super_res=False).to(device)
                        else:
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=32, num_classes=1, super_res=False).to(device)

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
                        pred_y = model(graph_data.x, graph_data.edge_index, bg_subgraph=graph_data.entity_ids).cpu().detach().numpy().squeeze(-1)
                        true_y = graph_data.y.cpu().detach().numpy().squeeze(-1)

                        train_r2 = r2_score(true_y[train_idx], pred_y[train_idx])
                        val_r2 = r2_score(true_y[val_idx], pred_y[val_idx])
                        test_r2 = r2_score(true_y[test_idx], pred_y[test_idx])

                        train_mse = mean_squared_error(true_y[train_idx], pred_y[train_idx])
                        val_mse = mean_squared_error(true_y[val_idx], pred_y[val_idx])
                        test_mse = mean_squared_error(true_y[test_idx], pred_y[test_idx])


                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['target_entity'] = target_entity
                        curr_results['num_datapoints'] = num_datapoints
                        curr_results['num_entities'] = graph_data.y.shape[0]
                        curr_results['model_embeddings'] = model_better_names[model_name]
                        curr_results['trial'] = trial
                        curr_results['train_r2'] = train_r2
                        curr_results['val_r2'] = val_r2
                        curr_results['test_r2'] = test_r2
                        curr_results['train_mse'] = train_mse
                        curr_results['val_mse'] = val_mse
                        curr_results['test_mse'] = test_mse
                        


                        total_results.append(curr_results.copy())

                    total_results_df = pd.DataFrame(total_results)

                    total_results_df.to_csv(args.results_csv)


if __name__ == "__main__":
    main()