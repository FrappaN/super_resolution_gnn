import torch

import datacommons_pandas as dc
import pandas as pd

# convert the df into a pytorch geometric data object
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, scatter

# create a kNN graph with k=10
from torch_geometric.nn import knn_graph
from torch_geometric import seed_everything

import numpy as np
from data_loader import *
from models import GNNPredictor
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

import lightgbm as lgb

from tqdm import tqdm
import argparse

import torch


from sklearn.metrics import roc_curve

def get_optimal_thresholds(y_true, y_probs, n_classes):
    """
    Finds the best threshold for each class in a multiclass problem.
    y_true: Ground truth labels (0 to N-1)
    y_probs: Predicted probabilities (shape: n_samples, n_classes)
    """
    best_thresholds = []
    
    for i in range(n_classes):
        # Create binary labels for the current class
        y_binary = (y_true == i).astype(int)
        # Extract probabilities for the current class
        y_score = y_probs[:, i]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_binary, y_score)
        
        # Calculate Youden's J statistic: J = tpr – fpr
        # The index of the maximum J gives the optimal threshold
        idx = np.argmax(tpr - fpr)
        best_thresholds.append(thresholds[idx])
        
    return best_thresholds


def create_spatial_graph(state_buss, buss_out_embs, macroentity, k=10, device='cpu'):
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

    train_y = torch.tensor(state_buss.train_target.values.astype(int), dtype=torch.long).view(-1)
    test_y = torch.tensor(state_buss.test_target.values.astype(int), dtype=torch.long).view(-1)

    block_groups = state_buss.block_group_id.values
    macroentities = state_buss[f'{macroentity}_id'].values

    # mapped np.array from block group id to integer
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

    # assign to each node its block group id as integer
    # same for its macroentity id

    train_labels = scatter(train_y, macroentity_ids, dim=0, reduce='mean').long()
    test_labels = scatter(test_y, block_group_ids, dim=0, reduce='mean').long()
    train_to_test_labels = scatter(train_y, block_group_ids, dim=0, reduce='mean').long()


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

def classify_predictions(true_y, continuous_pred_y, n_quantiles):
    """
    Classify the continuous predictions by first finding the best threshold for each class and then classifying accordingly.
    """
    # Get the optimal thresholds for each class
    thresholds = get_optimal_thresholds(true_y, continuous_pred_y, n_quantiles)

    rescaled_pred_y = np.zeros_like(continuous_pred_y)
    if n_quantiles == 1:
        preds = (continuous_pred_y[:, 1] >= thresholds[1]).astype(int)
    else:
        for i in range(n_quantiles):
            rescaled_pred_y[:, i] = continuous_pred_y[:, i] / thresholds[i]

        # Classify based on the thresholds
        preds = np.argmax(rescaled_pred_y, axis=1)

    return preds


def main(
    n_quantiles=5,
    num_trials=10,
    urban_areas=None,
    macroentities=None,
    business_csv_path='../datasets/yelp2019_business.csv',
    models_dir='../results/yelp2019/',
    results_csv='../results/yelp2019/super_resolution_GNN_classification_results.csv',
    seed=42,
    epochs=5000,
    patience=100,
    k=10,
    hidden_channels=32,
    lr=0.01,
    weight_decay=5e-4,
):
    if urban_areas is None:
        urban_areas = ['Philadelphia', 'St. Louis', 'Indianapolis', 'Nashville', 'Tampa']
    if macroentities is None:
        macroentities = ['zip', 'tract']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    business_df = pd.read_csv(business_csv_path)

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
    
    model_embs_names = {"NoEdges": "Emb",  "LightGCN": "LightGCN",
                      "FeatLightGCN": "Feat-LightGCN", "FeatAndPCLightGCN": "Feat\&PC-LightGCN", 
                      "NoFeatures": "RGNN", "FeatAndEmb": "Feat-RGNN", "FeatAndEmbAndPostalCode": "Feat\&PC-RGNN",
                        "NoPreTrain": "NoPreTrain"}

    dem_var_list = ['Median_Income', 'Median_HomeValue', 'Median_Age', ] #
    dem_var_names = {'Median_HomeValue': 'Median Home Value', 'Median_Income': 'Median Income', 'Median_Age': 'Median Age'}

    total_results = []
    for macroentity in macroentities: #
        full_bg_df = load_demographics(states_names_for_superres, block_groups)
        # select only block groups within Tracts with more than one block group
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
                
                # here we process the target variable
                curr_bg_values = full_bg_df.loc[curr_bg, dem_var_name].values

                if len(curr_bg_values) == 0:

                    curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_id', 'longitude', 'latitude']].copy()
                    curr_buss_df.loc[:, 'train_target'] = None
                    curr_buss_df.loc[:, 'test_target'] = None
                    continue

                curr_me_values = full_bg_df.loc[curr_bg, f'{dem_var_name}_{macroentity}'].values

                # only consider bg with non-null macroentity and block group values
                valid_bg = ~np.isnan(curr_me_values) & ~np.isnan(curr_bg_values)
                curr_bg = np.array(curr_bg)[valid_bg]
                curr_me_values = curr_me_values[valid_bg]
                curr_bg_values = curr_bg_values[valid_bg]

                #curr_unique_me_values = full_bg_df.loc[curr_bg].groupby(f'{macroentity}_id')[f'{dem_var_name}_{macroentity}'].mean().values

                full_bg_df.loc[curr_bg, f'binned_{dem_var_name}_{macroentity}'] = pd.qcut(curr_me_values, q=n_quantiles, labels=False, duplicates='drop')
                full_bg_df.loc[curr_bg, f'binned_{dem_var_name}'] = pd.qcut(curr_bg_values, q=n_quantiles, labels=False, duplicates='drop')
                full_bg_df.loc[curr_bg, f'ranking_{dem_var_name}_{macroentity}'] = pd.Series(curr_me_values).rank().values / len(curr_me_values)

                curr_buss_df = curr_buss_df[['block_group_id', f'{macroentity}_id', 'longitude', 'latitude']].copy()

                curr_buss_df.loc[:, 'train_target'] = full_bg_df.loc[curr_buss_bg, f'binned_{dem_var_name}_{macroentity}'].values

                curr_buss_df.loc[:, 'test_target'] = full_bg_df.loc[curr_buss_bg, f'binned_{dem_var_name}'].values
                curr_buss_df.loc[:, 'ranking_train'] = full_bg_df.loc[curr_buss_bg, f'ranking_{dem_var_name}_{macroentity}'].values

                curr_buss_df = curr_buss_df.dropna().copy()
                if len(curr_buss_df) == 0:
                    print(f'No businesses in urban area {urbe} with non-null target variable {dem_var_name}, skipping')
                    break

                for model_embs in model_embs_names.keys(): #
                    print(f'Running model {model_embs}, urban area {urbe}, target variable {dem_var_name}')
                    if model_embs == 'NoPreTrain':
                        bus__out_embeddings = torch.arange(len(curr_buss_df), dtype=torch.long).view(-1)
                    else:
                        bus__out_embeddings = np.load(f'{models_dir}{model_embs}__bus__out_embeddings.npy')
                        bus__out_embeddings = torch.tensor(bus__out_embeddings[curr_buss_df.index], dtype=torch.float)

                    graph_data = create_spatial_graph(curr_buss_df, bus__out_embeddings, macroentity, k=k, device=device)


                    assert torch.unique(graph_data.train_y).shape[0] == n_quantiles, f'Number of unique train_y {torch.unique(graph_data.train_y).shape[0]} does not match n_quantiles {n_quantiles}'
                    assert torch.unique(graph_data.test_y).shape[0] == n_quantiles, f'Number of unique test_y {torch.unique(graph_data.test_y).shape[0]} does not match n_quantiles {n_quantiles}'
                    seed_everything(seed)

                    curr_results = {}
                    # split train and validation
                    num_me_ids = len(curr_buss_df[f'{macroentity}_id'].unique())
                    assert num_me_ids == len(graph_data.train_y)

                    for trial in range(num_trials):

                        seed_everything(trial)
                        train_labels = graph_data.train_y
                        train_idx = []
                        val_idx = []
                        for class_id in range(n_quantiles):
                            class_indices = (train_labels == class_id).nonzero(as_tuple=True)[0]
                            # shuffle the indices
                            class_indices = class_indices[torch.randperm(len(class_indices))]
                            split = int(0.8 * len(class_indices))
                            train_idx.append(class_indices[:split])
                            val_idx.append(class_indices[split:])
                        train_idx = torch.cat(train_idx, dim=0).to('cpu')
                        val_idx = torch.cat(val_idx, dim=0).to('cpu')

                        # cross entropy loss with logits
                        criterion = torch.nn.CrossEntropyLoss()
                        if model_embs == 'NoPreTrain':
                            model = GNNPredictor(in_channels=32, hidden_channels=hidden_channels, num_classes=n_quantiles, learnable_emb=True,num_nodes=len(curr_buss_df)).to(device)
                        else:
                            model = GNNPredictor(in_channels=graph_data.num_features, hidden_channels=hidden_channels, num_classes=n_quantiles).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                        best_loss = float('inf')
                        count = 0
                        assert torch.unique(graph_data.train_y[train_idx]).shape[0] == n_quantiles, f'Number of unique train_y in train_idx {torch.unique(graph_data.train_y[train_idx]).shape[0]} does not match n_quantiles {n_quantiles}'
                        assert torch.unique(graph_data.train_y[val_idx]).shape[0] == n_quantiles, f'Number of unique train_y in val_idx {torch.unique(graph_data.train_y[val_idx]).shape[0]} does not match n_quantiles {n_quantiles}'
                        assert torch.unique(graph_data.test_y).shape[0] == n_quantiles, f'Number of unique test_y {torch.unique(graph_data.test_y).shape[0]} does not match n_quantiles {n_quantiles}'

                        model.train()
                        for epoch in (range(1, epochs + 1)):
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

                        pred_y_on_train =  model(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph, graph_data.me_subgraph).softmax(dim=1).detach().cpu().numpy()
                        pred_y_on_test = model.forward_bg(graph_data.x, graph_data.edge_index, graph_data.bg_subgraph).softmax(dim=1).detach().cpu().numpy()
                        val_y = graph_data.train_y[val_idx].cpu().detach().numpy()
                        train_y = graph_data.train_y[train_idx].cpu().detach().numpy()


                        train_score = f1_score(train_y, pred_y_on_train.argmax(axis=1)[train_idx.cpu().numpy()], average='micro')
                        val_score = f1_score(val_y, pred_y_on_train.argmax(axis=1)[val_idx.cpu().numpy()], average='micro')
                        if n_quantiles == 2:
                            train_auc = roc_auc_score(train_y, pred_y_on_train[train_idx, 1])
                            val_auc = roc_auc_score(val_y, pred_y_on_train[val_idx, 1])
                        else:
                            train_auc = roc_auc_score(train_y, pred_y_on_train[train_idx], multi_class='ovr')
                            val_auc = roc_auc_score(val_y, pred_y_on_train[val_idx], multi_class='ovr')

                        train_y = graph_data.train_to_test_y.cpu().detach().numpy()
                        test_y = graph_data.test_y.cpu().detach().numpy()
                        cont_train_y = curr_buss_df.groupby('block_group_id')['ranking_train'].mean().values

                        # concatenate n_quantiles times train_y (horizzontaly)
                        discrete_pred_y = pred_y_on_test.argmax(axis=1) #classify_predictions(test_y, pred_y_on_test, n_quantiles)

                        train_to_test_score = f1_score(test_y, train_y, average='micro')
                        if n_quantiles == 2:
                            train_to_test_auc = roc_auc_score(test_y, cont_train_y)
                        else:
                            one_hot_train_y = np.zeros((len(test_y), n_quantiles))
                            for i in range(n_quantiles):
                                one_hot_train_y[:, i] = (train_y == i).astype(int)
                            train_to_test_auc = roc_auc_score(test_y, one_hot_train_y, multi_class='ovr')

                        test_score = f1_score(test_y, discrete_pred_y, average='micro')
                        if n_quantiles == 2:
                            test_auc = roc_auc_score(test_y, pred_y_on_test[:, 1])
                        else:
                            test_auc = roc_auc_score(test_y, pred_y_on_test, multi_class='ovr')

                        curr_results['urban_area'] = urbe
                        curr_results['target_variable'] = dem_var_names[dem_var_name]
                        curr_results['macroentity'] = macroentity
                        curr_results['n_quantiles'] = n_quantiles
                        curr_results['num_datapoints'] = len(curr_buss_df)
                        curr_results['num_macroentities'] = len(curr_buss_df[f'{macroentity}_id'].unique())
                        curr_results['num_bg'] = len(curr_buss_df.block_group_id.unique())
                        curr_results['model_embeddings'] = model_embs_names[model_embs]
                        curr_results['trial'] = trial
                        curr_results['train_f1'] = train_score
                        curr_results['val_f1'] = val_score
                        curr_results['test_f1'] = test_score
                        curr_results['train_to_test_f1'] = train_to_test_score
                        curr_results['train_auc'] = train_auc
                        curr_results['val_auc'] = val_auc
                        curr_results['test_auc'] = test_auc
                        curr_results['train_to_test_auc'] = train_to_test_auc

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
    parser.add_argument("--results_csv", type=str, default='/super_resolution_GNN_classification_results.csv')
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