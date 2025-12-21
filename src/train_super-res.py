import torch

import datacommons_pandas as dc
import pandas as pd
import numpy as np

# convert the df into a pytorch geometric data object
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, scatter

# create a kNN graph with k=10
from torch_geometric.nn import knn_graph
from torch_geometric import seed_everything

from sklearn.preprocessing import QuantileTransformer
from data_loader import *
from models import GNNPredictor
from sklearn.metrics import r2_score, mean_absolute_error


import argparse

def create_spatial_graph(state_buss, buss_out_embs, macroentity, k=10, device='cpu'):
    """
    Build spatial graph and assign labels aggregated like the classification pipeline:
    - train_y: aggregated at macroentity level (mean over businesses)
    - test_y: aggregated at block_group level (mean over businesses)
    - train_to_test_y: macroentity-level train labels aggregated to block_group level
    Additionally expose train_subgraph/test_subgraph assignments for scatter aggregation during training/eval.
    """
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

    train_y = torch.tensor(state_buss.train_target.values, dtype=torch.float).view(-1, 1)
    test_y = torch.tensor(state_buss.test_target.values, dtype=torch.float).view(-1, 1)

    # map block groups and macroentities to integer ids
    block_groups = state_buss.block_group_id.values
    macroentities = state_buss[f'{macroentity}_id'].values
    _, block_groups = np.unique(block_groups, return_inverse=True)
    _, macroentities = np.unique(macroentities, return_inverse=True)
    block_group_ids = torch.tensor(block_groups, dtype=torch.long).view(-1)
    macroentity_ids = torch.tensor(macroentities, dtype=torch.long).view(-1)

    # create a mapping from the block group id to its macroentity id
    blockgroup_to_macroentity = {}
    for bg_id, me_id in zip(block_group_ids.numpy(), macroentity_ids.numpy()):
        blockgroup_to_macroentity[bg_id] = me_id
    # now we can create a tensor that maps each block group id to its macroentity id
    blockgroup_to_macroentity = dict(sorted(blockgroup_to_macroentity.items()))
    blockgroup_to_macroentity_tensor = torch.tensor(list(blockgroup_to_macroentity.values()), dtype=torch.long)
    
    # aggregate labels
    train_labels = scatter(train_y, macroentity_ids, dim=0, reduce='mean')  # (num_macro, 1)
    test_labels = scatter(test_y, block_group_ids, dim=0, reduce='mean')    # (num_bg, 1)
    train_to_test_labels = scatter(train_y, block_group_ids, dim=0, reduce='mean')

    data = Data(
        x=buss_out_embs, 
        edge_index=edge_index, 
        train_y=train_labels, 
        test_y=test_labels, 
        train_to_test_y=train_to_test_labels, 
        me_subgraph=blockgroup_to_macroentity_tensor, 
        bg_subgraph=block_group_ids
    ).to(device)

    return data

    

def main(
    n_quantiles=5,
    num_trials=10,
    urban_areas=None,
    macroentities=None,
    business_csv_path='../datasets/yelp2019_business.csv',
    models_dir='../results/yelp2019/',
    results_csv='../results/yelp2019/super_resolution_GNN_results.csv',
    seed=42,
    epochs=5000,
    patience=100,
    k=10,
    hidden_channels=32,
    lr=0.01,
    weight_decay=5e-4,
):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    business_df = pd.read_csv(business_csv_path)

    business_df.loc[:, 'postal_code'] = business_df['postal_code'].fillna(0)
    business_df['postal_code'] = business_df['postal_code'].astype(str)
    business_df.loc[:, 'block_group_id'] = business_df['block_group_id'].fillna(0)
    business_df['block_group_id'] = business_df['block_group_id'].astype(int).astype(str)

    block_groups = business_df['block_group_id'].unique()

    if urban_areas is None:
        urban_areas = ['Philadelphia', 'St. Louis', 'Indianapolis', 'Nashville', 'Tampa']
    if macroentities is None:
        macroentities = ['zip', 'tract']

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

    model_embs_names = {"NoEdges": "Emb",  "LightGCN": "LightGCN",
                        "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN", 
                        "NoFeatures": "RGNN", "FeatAndEmb": "Feat-RGNN", "FeatAndEmbAndPostalCode": "Feat\&PC-RGNN",
                        "NoPreTrain": "NoPreTrain"}

    dem_var_list = ['Median_Income', 'Median_HomeValue', 'Median_Age']
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}

    total_results = []

    for macroentity in macroentities:
        full_bg_df = load_demographics(states_names_for_superres, block_groups)
        # select only block groups within macroentities with more than one block group
        full_bg_df = full_bg_df.groupby(macroentity).filter(lambda x: len(x) > 1)

        for urbe in urban_areas:
            for dem_var_name in dem_var_list:
                curr_buss = business_df.urban_area == urbe
                curr_buss_df = business_df.loc[curr_buss].copy()
                curr_bg = curr_buss_df.block_group_id.unique()
                # check that all the curr_bg are in full_bg_df
                curr_bg = [bg for bg in curr_bg if bg in full_bg_df.index]
                curr_buss = business_df.block_group_id.isin(curr_bg)
                curr_buss_df = business_df.loc[curr_buss].copy()

                # drop any business whose block group target variable is null
                curr_buss_df = curr_buss_df[~curr_buss_df.block_group_id.isin(full_bg_df[full_bg_df[dem_var_name].isnull()].index)]

                # similarly, drop any business whose macroentity target variable is null
                curr_buss_df = curr_buss_df[~curr_buss_df.block_group_id.isin(full_bg_df[full_bg_df[f'{dem_var_name}_{macroentity}'].isnull()].index)]

                curr_buss_bg = curr_buss_df.block_group_id.values
                curr_buss_df[f'{macroentity}_id'] = full_bg_df.loc[curr_buss_bg, f'{macroentity}_dcid'].values

                # process continuous targets for regression (scaled for stability)
                curr_bg_values = full_bg_df.loc[curr_bg, dem_var_name].values

                if len(curr_bg_values) == 0:
                    curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_id', 'longitude', 'latitude']].copy()
                    curr_buss_df.loc[:, 'train_target'] = None
                    curr_buss_df.loc[:, 'test_target'] = None
                    continue

                curr_me_values = full_bg_df.loc[curr_bg, f'{dem_var_name}_{macroentity}'].values
                curr_unique_me_values = full_bg_df.loc[curr_bg].groupby(f'{macroentity}_dcid')[f'{dem_var_name}_{macroentity}'].mean().values

                target_scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution='normal')
                target_scaler.fit(curr_unique_me_values.reshape(-1, 1))
                full_bg_df.loc[curr_bg, f'scaled_{dem_var_name}_{macroentity}'] = target_scaler.transform(curr_me_values.reshape(-1, 1))
                full_bg_df.loc[curr_bg, f'scaled_{dem_var_name}'] = target_scaler.transform(curr_bg_values.reshape(-1, 1))

                curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_id', 'longitude', 'latitude']].copy()

                curr_buss_df.loc[:, 'train_target'] = full_bg_df.loc[curr_buss_bg, f'scaled_{dem_var_name}_{macroentity}'].values
                curr_buss_df.loc[:, 'test_target'] = full_bg_df.loc[curr_buss_bg, f'scaled_{dem_var_name}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                for model_embs in model_embs_names.keys():
                    print(f'Running model {model_embs}, urban area {urbe}, target variable {dem_var_name}')
                    if model_embs == 'NoPreTrain':
                        bus__out_embeddings = torch.arange(len(curr_buss_df)).view(-1)
                    else:
                        bus__out_embeddings = np.load(f'{models_dir}{model_embs}__bus__out_embeddings.npy')
                        bus__out_embeddings = torch.tensor(bus__out_embeddings[curr_buss_df.index])

                    graph_data = create_spatial_graph(curr_buss_df, bus__out_embeddings, macroentity, k=k, device=device)

                    seed_everything(seed)

                    curr_results = {}
                    # split train and validation on macroentities (train_y length)
                    num_train_ids = len(graph_data.train_y)


                    perm = torch.randperm(num_train_ids)
                    train_idx = perm[:int(0.8 * num_train_ids)].to('cpu')
                    val_idx = perm[int(0.8 * num_train_ids):].to('cpu')

                    for trial in range(num_trials):
                        seed_everything(trial)
                        # learn the model
                        criterion = torch.nn.MSELoss()
                        if model_embs == 'NoPreTrain':
                            model = GNNPredictor(in_channels=32, hidden_channels=hidden_channels,num_classes=1, learnable_emb=True,num_nodes=len(curr_buss_df)).to(device)
                        else:
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=hidden_channels, num_classes=1).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                        best_loss = float('inf')
                        count = 0

                        model.train()
                        for epoch in range(1, epochs + 1):
                            model.train()
                            optimizer.zero_grad()
                            out = model(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph, graph_data.me_subgraph)
                            loss = criterion(out[train_idx], graph_data.train_y[train_idx])
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            out = model(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph, graph_data.me_subgraph)
                            val_loss = criterion(out[val_idx], graph_data.train_y[val_idx])
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_model = model.state_dict()
                                count = 0
                            else:
                                count += 1
                            if count >= patience:
                                break
                        model.load_state_dict(best_model)

                        model.eval()

                        pred_y_on_train = model(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph, graph_data.me_subgraph).reshape(-1).cpu().detach().numpy()
                        pred_y_on_test = model.forward_bg(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph).reshape(-1).cpu().detach().numpy()

                        val_y = graph_data.train_y[val_idx].cpu().detach().numpy().reshape(-1)
                        train_y = graph_data.train_y[train_idx].cpu().detach().numpy().reshape(-1)
                        
                        train_r2 = r2_score(train_y, pred_y_on_train[train_idx.cpu().numpy()])
                        train_mae = mean_absolute_error(train_y, pred_y_on_train[train_idx.cpu().numpy()])

                        val_r2 = r2_score(val_y, pred_y_on_train[val_idx.cpu().numpy()])
                        val_mae = mean_absolute_error(val_y, pred_y_on_train[val_idx.cpu().numpy()])

                        # Evaluate train→test and test metrics at block-group level
                        test_y = graph_data.test_y.cpu().detach().numpy().reshape(-1)
                        train_to_test_y = graph_data.train_to_test_y.cpu().detach().numpy().reshape(-1)

                        train_to_test_r2 = r2_score(test_y, train_to_test_y)
                        train_to_test_mae = mean_absolute_error(test_y, train_to_test_y)
                        test_r2 = r2_score(test_y, pred_y_on_test)
                        test_mae = mean_absolute_error(test_y, pred_y_on_test)

                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['macroentity'] = macroentity
                        curr_results['num_datapoints'] = len(curr_buss_df)
                        curr_results['num_macroentities'] = len(curr_buss_df[f'{macroentity}_id'].unique())
                        curr_results['num_bg'] = len(curr_buss_df.block_group_id.unique())
                        curr_results['model_embeddings'] = model_embs_names[model_embs]
                        curr_results['trial'] = trial
                        curr_results['train_r2'] = train_r2
                        curr_results['val_r2'] = val_r2
                        curr_results['test_r2'] = test_r2
                        curr_results['train_to_test_r2'] = train_to_test_r2
                        curr_results['train_mae'] = train_mae
                        curr_results['val_mae'] = val_mae
                        curr_results['test_mae'] = test_mae
                        curr_results['train_to_test_mae'] = train_to_test_mae

                        total_results.append(curr_results.copy())

                        total_results_df = pd.DataFrame(total_results)
                        total_results_df.to_csv(results_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_quantiles", type=int, default=5)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--urban_areas", type=str, default=None, help="Comma separated list, e.g. 'Philadelphia,St. Louis'")
    parser.add_argument("--macroentities", type=str, default=None, help="Comma separated list, e.g. 'zip,tract'")
    parser.add_argument("--business_csv_path", type=str, default='../datasets/yelp2019_business.csv')
    parser.add_argument("--models_dir", type=str, default='../results/yelp2019/')
    parser.add_argument("--results_path", type=str, default='../results/yelp2019/')
    parser.add_argument("--results_csv", type=str, default='super_resolution_GNN_results.csv')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)

    args = parser.parse_args()
    ua = args.urban_areas.split(",") if args.urban_areas else None
    me = args.macroentities.split(",") if args.macroentities else None

    results_csv_path = args.results_path + args.results_csv

    main(
        n_quantiles=args.n_quantiles,
        num_trials=args.num_trials,
        urban_areas=ua,
        macroentities=me,
        business_csv_path=args.business_csv_path,
        models_dir=args.models_dir,
        results_csv=results_csv_path,
        seed=args.seed,
        epochs=args.epochs,
        patience=args.patience,
        k=args.k,
        hidden_channels=args.hidden_channels,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )