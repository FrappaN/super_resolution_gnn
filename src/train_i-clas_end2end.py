import torch


import datacommons_pandas as dc
import pandas as pd

# convert the df into a pytorch geometric data object
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph

# create a kNN graph with k=10
from torch_geometric.nn import knn_graph
from torch_geometric import seed_everything


from data_loader import *
from models import GNNClassifier
from models_LightGCN import LightGCN, FeatLightGCN, FeatAndPCLightGCN, LightGCNRegression
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

from tqdm import tqdm
import numpy as np

def create_spatial_graph(state_buss, k=10, device='cpu'):
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

    train_y = torch.tensor(state_buss.target.values.astype(float), dtype=torch.float).view(-1, 1)

    return edge_index.to(device), train_y.to(device)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yelp_manager = Yelp_data_manager('yelp2019')


    business_df = pd.read_csv(f'../datasets/yelp2019_business.csv')

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
    
    
    model_better_names = {"LightGCN": "LightGCN", "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN"}

    
    dem_var_list = [ 'Median_Income', 'Median_HomeValue', 'Commute_Time', 'Median_Age', ] #
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Commute_Time': 'Commute Time', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}

    urban_areas = ['Philadelphia', 'St. Louis', 'Indianapolis', 'Nashville', 'Tampa']

    full_bg_df = load_demographics(states_names_for_superres, block_groups)

    total_results = []
    for target_entity in ['bg', 'tract', 'zip']:
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
                    curr_buss_df.loc[:, 'entity_dcid'] = curr_buss_df.block_group_id.values
                else:
                    curr_buss_df.loc[:, 'entity_dcid'] = full_bg_df.loc[curr_buss_bg, f'{target_entity}_dcid'].values

                if len(curr_values) == 0:
                    curr_buss_df = curr_buss_df[['entity_dcid', 'longitude', 'latitude',]].copy()
                    curr_buss_df.loc[:, 'target'] = None
                    continue


                full_bg_df.loc[curr_bg, f'binarized_{dem_var_name}'] = (curr_values >= np.nanmedian(curr_unique_values)).astype(int)

                curr_buss_df = curr_buss_df[['entity_dcid', 'longitude', 'latitude', ]].copy()

                curr_buss_df.loc[:, 'target'] = full_bg_df.loc[curr_buss_bg, f'binarized_{dem_var_name}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                num_datapoints = len(curr_buss_df)

                for model_name in model_better_names.keys(): #
                    print(f'Running model {model_name}, urban area {urbe}, target variable {dem_var_name}')
                    if model_name == 'Random':
                        bus__out_embeddings = torch.nn.Embedding(len(curr_buss_df), 32).weight.detach().cpu().numpy()
                    else:
                        bus__out_embeddings = np.load(f'../results/yelp2019/{model_name}__bus__out_embeddings.npy')
                        bus__out_embeddings = bus__out_embeddings[curr_buss_df.index]

                    curr_results = {}
                    # split businesses into train, val, test according to their entity_dcid
                    labels_per_entity = curr_buss_df.groupby('entity_dcid')['target'].first()

                    pos_entities = labels_per_entity[labels_per_entity == 1].index.values
                    neg_entities = labels_per_entity[labels_per_entity == 0].index.values

                    for trial in range(num_trials):

                        yelp_data = Yelp_data_manager(version='yelp2019', seed=trial, heterogenous=False)
                        data = yelp_data.get_data()
                        # restrict data to curr_buss_df
                        buss_indices = curr_buss_df.index.values
                        buss_indices_tensor = torch.tensor(buss_indices, dtype=torch.long)
                        buss_indices_tensor += data.num_users  # shift by num_users to get the correct indices in the full graph
                        subset, edge_index_user_business, mapping, _ = k_hop_subgraph(buss_indices_tensor, 1, data.edge_index, relabel_nodes=True)
                        edge_index, all_y = create_spatial_graph(curr_buss_df, k=10, device=device)
                        edge_index_user_business = edge_index_user_business.to(device)
                        num_users = torch.sum(subset < data.num_users).item()
                        num_businesses = len(subset) - num_users
                        assert num_businesses == len(curr_buss_df)
                        num_nodes = torch.max(edge_index_user_business).item() + 1
                        x = data.x[buss_indices_tensor-data.num_users].to(device)
                        pcs = data.pc[buss_indices_tensor-data.num_users].to(device)
                        _, pcs = torch.unique(pcs, return_inverse=True)

                        if model_name == 'LightGCN':
                                model = LightGCNRegression(
                                    encoder_class=LightGCN,
                                    num_nodes=num_nodes,
                                    embedding_dim=64,
                                    num_layers=2,
                                    in_features=data.x.size(1),
                                    alpha=1/3,
                                    num_users=num_users,
                                    num_pcs=len(pcs.unique()),
                                )
                        elif model_name == 'FeatLightGCN':
                            model = LightGCNRegression(
                                encoder_class=FeatLightGCN,
                                num_nodes=num_nodes,
                                embedding_dim=64,
                                num_layers=2,
                                in_features=data.x.size(1),
                                alpha=1/3,
                                num_users=num_users,
                                num_pcs=len(pcs.unique()),
                            )
                        elif model_name == 'FeatAndPCLightGCN':
                            model = LightGCNRegression(
                                encoder_class=FeatAndPCLightGCN,
                                num_nodes=num_nodes,
                                embedding_dim=64,
                                num_layers=2,
                                in_features=data.x.size(1),
                                alpha=1/3,
                                num_users=num_users,
                                num_pcs=len(pcs.unique()),
                            )
                        model = model.to(device)

                        # learn the model on the urban area
                        seed_everything(trial)
                        np.random.shuffle(pos_entities)
                        np.random.shuffle(neg_entities)

                        # select entities such that train, val and test are balanced
                        train_ents = np.concatenate([pos_entities[:int(0.6*len(pos_entities))], 
                                                    neg_entities[:int(0.6*len(neg_entities))]])
                        val_ents = np.concatenate([pos_entities[int(0.6*len(pos_entities)):int(0.8*len(pos_entities))], 
                                                neg_entities[int(0.6*len(neg_entities)):int(0.8*len(neg_entities))]])
                        test_ents = np.concatenate([pos_entities[int(0.8*len(pos_entities)):], 
                                                    neg_entities[int(0.8*len(neg_entities)):]])

                        train_idx = curr_buss_df.entity_dcid.isin(train_ents).to_numpy()
                        val_idx = curr_buss_df.entity_dcid.isin(val_ents).to_numpy()
                        test_idx = curr_buss_df.entity_dcid.isin(test_ents).to_numpy()

                        all_idx = np.arange(len(curr_buss_df))

                        train_idx = torch.tensor(all_idx[train_idx], dtype=torch.long)
                        val_idx = torch.tensor(all_idx[val_idx], dtype=torch.long)
                        test_idx = torch.tensor(all_idx[test_idx], dtype=torch.long)

                        if len(train_idx) == 0 or len(test_idx) == 0 or len(val_idx) == 0:
                            if len(train_idx) == 0:
                                print(f'No train businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            if len(test_idx) == 0:
                                print(f'No test businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            if len(val_idx) == 0:
                                print(f'No val businesses in urban area {urbe} with non-null target variable {dem_var_name} on {target_entity}, skipping')
                            continue
                        criterion = torch.nn.BCEWithLogitsLoss()

                        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                        best_loss = float('inf')
                        count = 0

                        model.train()
                        for epoch in (range(1, 5001)):
                            model.train()
                            optimizer.zero_grad()
                            out = model(edge_index_user_business=edge_index_user_business, edge_index=edge_index, x_bus=x, pcs=pcs)
                            loss = criterion(out[train_idx], all_y[train_idx])
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            out = model(edge_index_user_business=edge_index_user_business, edge_index=edge_index, x_bus=x, pcs=pcs)
                            val_loss = criterion(out[val_idx], all_y[val_idx])
                            if val_loss < best_loss:
                                best_loss = val_loss
                                best_model = model.state_dict()
                                count = 0
                            else:
                                count += 1
                            if count >= 100:
                                break
                        model.load_state_dict(best_model)

                        model.eval()


                        pred_y = model(edge_index_user_business=edge_index_user_business, edge_index=edge_index, x_bus=x, pcs=pcs).sigmoid().cpu().detach().numpy()
                        # else:
                        # model = lgb.LGBMRegressor(random_state=trial)
                        # train_X = bus__out_embeddings[train_idx]
                        # train_y = curr_buss_df.loc[train_idx, 'target'].values
                        # val_X = bus__out_embeddings[val_idx]
                        # val_y = curr_buss_df.loc[val_idx, 'target'].values
                        # test_X = bus__out_embeddings[test_idx]
                        # test_y = curr_buss_df.loc[test_idx, 'target'].values
                        # model.fit(train_X, train_y, eval_set=[(val_X, val_y)])

                        # pred_y = model.predict_proba(bus__out_embeddings)[:, 1]
                        # val_score = roc_auc_score(val_y, pred_y[val_idx])


                        # when using geoentity graph
                        # test_graph_data = create_geoentity_graph(curr_buss_df, bus__out_embeddings, geoentity='block_group_id', device=device)
                        # test_y = test_graph_data.test_y.cpu().detach().numpy()
                        # test_pred_y = model(test_graph_data.x, test_graph_data.edge_index).cpu().detach().numpy()
                        # test_score = mean_absolute_error(test_y, test_pred_y)

                        # train_y = test_graph_data.train_y.cpu().detach().numpy()


                        # when using spatial or covisit graph
                        # consider the prediction for each entity as the mean of the predictions of the businesses in the entity
                        curr_buss_df['pred_y'] = pred_y
                        train_y_and_test_y = curr_buss_df.groupby('entity_dcid')[['target', 'pred_y']].median()
                        true_y = train_y_and_test_y['target'].to_numpy()
                        pred_y = train_y_and_test_y['pred_y'].to_numpy()

                        test_y = true_y[train_y_and_test_y.index.isin(test_ents)]
                        test_pred_y = pred_y[train_y_and_test_y.index.isin(test_ents)]
                        test_score = roc_auc_score(test_y, test_pred_y)

                        train_y = true_y[train_y_and_test_y.index.isin(train_ents)]
                        train_pred_y = pred_y[train_y_and_test_y.index.isin(train_ents)]
                        train_score = roc_auc_score(train_y, train_pred_y)

                        val_y = true_y[train_y_and_test_y.index.isin(val_ents)]
                        val_pred_y = pred_y[train_y_and_test_y.index.isin(val_ents)]
                        val_score = roc_auc_score(val_y, val_pred_y)


                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['target_entity'] = target_entity
                        curr_results['num_datapoints'] = num_datapoints
                        curr_results['num_entities'] = len(pos_entities) + len(neg_entities)
                        curr_results['model_embeddings'] = model_better_names[model_name]
                        curr_results['trial'] = trial
                        curr_results['val_score'] = val_score

                        curr_results['test_score'] = test_score
                        curr_results['train_score'] = train_score
                        curr_results['val_score'] = val_score

                        total_results.append(curr_results.copy())

                    total_results_df = pd.DataFrame(total_results)

                    total_results_df.to_csv(f'../results/yelp2019/interpolation_E2E_results_classification.csv')


if __name__ == "__main__":
    main()