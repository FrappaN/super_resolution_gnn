import torch

import datacommons_pandas as dc
import pandas as pd

# convert the df into a pytorch geometric data object
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph

# create a kNN graph with k=10
from torch_geometric.nn import knn_graph
from torch_geometric import seed_everything


from sklearn.preprocessing import QuantileTransformer
from data_loader import *
from models import GNNClassifier
from models_LightGCN import LightGCN, FeatLightGCN, FeatAndPCLightGCN, LightGCNRegression
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from tqdm import tqdm

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

    train_y = torch.tensor(state_buss.train_target.values.astype(float), dtype=torch.float).view(-1, 1)
    test_y = torch.tensor(state_buss.test_target.values.astype(float), dtype=torch.float).view(-1, 1)
    
    return edge_index.to(device), train_y.to(device), test_y.to(device)


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
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
    
    model_better_names = { "LightGCN": "LightGCN", "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN"} # 

    dem_var_list = ['Median_Income', 'Median_HomeValue', 'Median_Age', ] #
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}

    urban_areas = ['Philadelphia', 'St. Louis', 'Indianapolis', 'Nashville', 'Tampa']

    total_results = []
    for macroentity in ['zip','tract' ]: #
        full_bg_df = load_demographics(states_names_for_superres, block_groups)
        # select only block groups within Tracts with more than one block group
        full_bg_df = full_bg_df.groupby(macroentity).filter(lambda x: len(x) > 1)

        
        num_trials = 10

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
                curr_buss_df[f'{macroentity}_dcid'] = full_bg_df.loc[curr_buss_bg, f'{macroentity}_dcid'].values
                
                # here we process the target variable
                curr_bg_values = full_bg_df.loc[curr_bg, dem_var_name].values

                if len(curr_bg_values) == 0:

                    curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_dcid', 'longitude', 'latitude']].copy()
                    curr_buss_df.loc[:, 'train_target'] = None
                    curr_buss_df.loc[:, 'test_target'] = None
                    continue

                curr_me_values = full_bg_df.loc[curr_bg, f'{dem_var_name}_{macroentity}'].values

                curr_unique_me_values = full_bg_df.loc[curr_bg].groupby(f'{macroentity}_dcid')[f'{dem_var_name}_{macroentity}'].mean().values

                full_bg_df.loc[curr_bg, f'binarized_{dem_var_name}_{macroentity}'] = curr_me_values >= np.nanmedian(curr_unique_me_values)
                full_bg_df.loc[curr_bg, f'binarized_{dem_var_name}'] = curr_bg_values >= np.nanmedian(curr_unique_me_values)


                curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_dcid', 'longitude', 'latitude']].copy()

                curr_buss_df.loc[:, 'train_target'] = full_bg_df.loc[curr_buss_bg, f'binarized_{dem_var_name}_{macroentity}'].values
                curr_buss_df.loc[:, 'test_target'] = full_bg_df.loc[curr_buss_bg, f'binarized_{dem_var_name}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                for model_name in model_better_names.keys(): #
                    print(f'Running model {model_name}, urban area {urbe}, target variable {dem_var_name}')


                    curr_results = {}

                    for trial in range(num_trials):

                        yelp_data = Yelp_data_manager(version='yelp2019', seed=trial, heterogenous=False)
                        data = yelp_data.get_data()
                        # restrict data to curr_buss_df
                        buss_indices = curr_buss_df.index.values
                        buss_indices_tensor = torch.tensor(buss_indices, dtype=torch.long)
                        buss_indices_tensor += data.num_users  # shift by num_users to get the correct indices in the full graph
                        subset, edge_index_user_business, mapping, _ = k_hop_subgraph(buss_indices_tensor, 1, data.edge_index, relabel_nodes=True)
                        edge_index, train_y, test_y = create_spatial_graph(curr_buss_df, k=10, device=device)
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

                        seed_everything(trial)
                        perm = torch.randperm(num_businesses)
                        train_idx = perm[:int(0.8 * num_businesses)]
                        val_idx = perm[int(0.8 * num_businesses):]

                        
                        criterion = torch.nn.BCEWithLogitsLoss()
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
                        best_loss = float('inf')
                        count = 0

                        model.train()
                        for epoch in (range(1, 5001)):
                            model.train()
                            optimizer.zero_grad()
                            out = model(edge_index_user_business=edge_index_user_business, edge_index=edge_index, x_bus=x, pcs=pcs)
                            loss = criterion(out[train_idx], train_y[train_idx])
                            loss.backward()
                            optimizer.step()

                            model.eval()
                            out = model(edge_index_user_business=edge_index_user_business, edge_index=edge_index, x_bus=x, pcs=pcs)
                            val_loss = criterion(out[val_idx], train_y[val_idx])
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
                        train_y = train_y.cpu().detach().numpy()

                        val_roc = roc_auc_score(train_y[val_idx], pred_y[val_idx])
                        # else:
                        # model = lgb.LGBMClassifier(random_state=trial)
                        # train_y = curr_buss_df.loc[:, 'train_target'].values.astype(bool)
                        # train_X = bus__out_embeddings
                        # # split train and validation
                        # num_nodes = len(curr_buss_df)
                        # perm = np.random.permutation(num_nodes)
                        # train_idx = perm[:int(0.8 * num_nodes)]
                        # val_idx = perm[int(0.8 * num_nodes):]
                        # val_X = bus__out_embeddings[val_idx]
                        # val_y = curr_buss_df.iloc[val_idx].loc[:,'train_target'].values.astype(bool)
                        # model.fit(train_X[train_idx], train_y[train_idx], eval_set=[(val_X, val_y)])

                        # pred_y = model.predict_proba(bus__out_embeddings)[:, 1]
                        # val_roc = roc_auc_score(val_y, pred_y[val_idx])


                        curr_buss_df.loc[:, 'pred_y'] = pred_y
                        aggregated_buss = curr_buss_df.groupby(f'{macroentity}_dcid')[['train_target', 'pred_y']].median()
                        train_y = aggregated_buss['train_target'].to_numpy().astype(bool)
                        pred_y = aggregated_buss['pred_y'].to_numpy()
                        
                        train_score = roc_auc_score(train_y, pred_y)

                        # consider the prediction for each block group as the mean of the predictions of the businesses in the block group
                        train_y_and_test_y = curr_buss_df.groupby('block_group_id')[['test_target', 'train_target', 'pred_y']].median()
                        test_y = train_y_and_test_y['test_target'].to_numpy().astype(bool)
                        train_y = train_y_and_test_y['train_target'].to_numpy().astype(bool)
                        pred_y = train_y_and_test_y['pred_y'].to_numpy()

                        train_to_test_score = roc_auc_score(test_y, train_y)
                        test_score = roc_auc_score(test_y, pred_y)

                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['macroentity'] = macroentity
                        curr_results['num_datapoints'] = len(curr_buss_df)
                        curr_results['num_macroentities'] = len(curr_buss_df[f'{macroentity}_dcid'].unique())
                        curr_results['num_bg'] = len(curr_buss_df.block_group_id.unique())
                        curr_results['model_embeddings'] = model_better_names[model_name]
                        curr_results['trial'] = trial
                        curr_results['val_ROCAUC'] = val_roc
                        curr_results[f'train_ROCAUC'] = train_score
                        curr_results[f'test_ROCAUC'] = test_score
                        curr_results[f'train_to_test_ROCAUC'] = train_to_test_score
                        
                        total_results.append(curr_results.copy())

                    total_results_df = pd.DataFrame(total_results)

                    total_results_df.to_csv(f'../results/yelp2019/super_resolution_E2E_classification_results.csv')


if __name__ == "__main__":
    main()