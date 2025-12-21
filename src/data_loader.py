import os
import torch
from torch_geometric import seed_everything
import pandas as pd
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as T
import numpy as np
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
import torch_geometric.utils as tgu
from torch_geometric import EdgeIndex

import datacommons_pandas as dc


class Yelp_data_manager:
    def __init__(self,version, with_attr=False, seed=180124, heterogenous=True):
        self.version = version
        self.seed = seed
        self.hetero = heterogenous
        self.with_attr = with_attr
        self._load_data(self.version)
        return
    
    def get_data(self):
        return self.data
    
    def _load_data(self, version, *args: torch.Any, **kwds: torch.Any) -> Data:

        seed_everything(self.seed)
        if self.with_attr:
            version = version + '_with_edge_attr'

        data = torch.load(f'../datasets/{version}.pt')

        pc_unique_values = data['business'].pc.unique().cpu().numpy()
        bg_unique_values = data['business'].bg.unique().cpu().numpy()
        np.random.shuffle(pc_unique_values)
        np.random.shuffle(bg_unique_values)

        postal_code_map = pd.Series(np.arange(len(pc_unique_values)), index=pc_unique_values, ) 
        postal_codes = postal_code_map[data['business'].pc.cpu().numpy()].values

        bg_map = pd.Series(np.arange(len(bg_unique_values)), index=bg_unique_values, )
        business_groups = bg_map[data['business'].bg.cpu().numpy()].values

        self.unique_values = pc_unique_values
        self.postal_code_map = postal_code_map
        data["business"].pc =  torch.from_numpy(postal_codes).to(torch.int64)
        data["business"].bg = torch.from_numpy(business_groups).to(torch.int64)
        
        if self.hetero:
            self.data = data
            self.sparse_size = (self.data['user'].num_nodes, self.data['business'].num_nodes)
        else:
            num_users = data['user'].num_nodes
            num_businesses = data['business'].num_nodes
            self.data = data.to_homogeneous()
            self.sparse_size = (num_users,num_businesses)
            self.data.pc = torch.from_numpy(postal_codes).to(torch.int64)
            self.data.bg = torch.from_numpy(business_groups).to(torch.int64)
            self.data.x = data['business'].x
            self.data.num_users = num_users
            self.data.num_businesses = num_businesses
            self.data.num_nodes = num_users + num_businesses
        return
    
    def set_data_split(self, seed=None):
        if seed is None:
            seed = self.seed
        seed_everything(seed)
        heterog_str = 'heterogenous' if self.hetero else 'homogeneous'

        # check if there is a file '../datasets/{self.version}_train_split_{seed}.pt'
        try:
            train_indices = torch.load(f'../datasets/{self.version}_{heterog_str}_train_split_{seed}.pt')
            val_indices = torch.load(f'../datasets/{self.version}_{heterog_str}_val_split_{seed}.pt')
            test_indices = torch.load(f'../datasets/{self.version}_{heterog_str}_test_split_{seed}.pt')
            if self.hetero:
                edge_label_index = self.data["user", "rates", "business"].edge_index
            else:
                edge_label_index = self.data.edge_index
                edge_weights = self.data.edge_weight
        except FileNotFoundError:
            if self.hetero:
                edge_index = self.data["user", "rates", "business"].edge_index
                all_user_train_indices = []
                all_user_val_indices = []
                all_user_test_indices = []
                # randomly select 0.8 of the edges of each user as training edges
                for user in range(self.data["user"].num_nodes):
                    mask = edge_index[0] == user
                    user_edges_indices = torch.nonzero(mask)

                    perm = torch.randperm(user_edges_indices.size(0))

                    user_train_perm_slice = perm[:int(0.8 * user_edges_indices.size(0))]
                    user_val_perm_slice = perm[int(0.8 * user_edges_indices.size(0)):int(0.9 * user_edges_indices.size(0))]
                    user_test_perm_slice = perm[int(0.9 * user_edges_indices.size(0)):]

                    all_user_train_indices.append(user_edges_indices[user_train_perm_slice])
                    all_user_val_indices.append(user_edges_indices[user_val_perm_slice])
                    all_user_test_indices.append(user_edges_indices[user_test_perm_slice])

                edge_label_index = self.data['user', 'rates', 'business'].edge_index
            else:
                edge_index = self.data.edge_index
                all_user_train_indices = []
                all_user_val_indices = []
                all_user_test_indices = []
                for user in range(self.data.num_users):
                    mask = edge_index[0] == user
                    user_edges_indices = torch.nonzero(mask)

                    perm = torch.randperm(user_edges_indices.size(0))

                    user_train_perm_slice = perm[:int(0.8 * user_edges_indices.size(0))]
                    user_val_perm_slice = perm[int(0.8 * user_edges_indices.size(0)):int(0.9 * user_edges_indices.size(0))]
                    user_test_perm_slice = perm[int(0.9 * user_edges_indices.size(0)):]

                    all_user_train_indices.append(user_edges_indices[user_train_perm_slice])
                    all_user_val_indices.append(user_edges_indices[user_val_perm_slice])
                    all_user_test_indices.append(user_edges_indices[user_test_perm_slice])

                edge_label_index = self.data.edge_index

            train_indices = torch.cat(all_user_train_indices, dim=0).squeeze()
            val_indices = torch.cat(all_user_val_indices, dim=0).squeeze()
            test_indices = torch.cat(all_user_test_indices, dim=0).squeeze()
            # save the splits
            
            torch.save(train_indices, f'../datasets/{self.version}_{heterog_str}_train_split_{seed}.pt')
            torch.save(val_indices, f'../datasets/{self.version}_{heterog_str}_val_split_{seed}.pt')
            torch.save(test_indices, f'../datasets/{self.version}_{heterog_str}_test_split_{seed}.pt')

        self.train_edge_label_index = EdgeIndex(
            edge_label_index[:, train_indices],
            sparse_size=self.sparse_size,
        ).sort_by('row')[0]
        self.val_edge_label_index = EdgeIndex(
            edge_label_index[:, val_indices],
            sparse_size=self.sparse_size,
        ).sort_by('row')[0]
        self.test_edge_label_index = EdgeIndex(
            edge_label_index[:, test_indices],
            sparse_size=self.sparse_size,
        ).sort_by('row')[0]
        self.train_val_edge_label_index = EdgeIndex(
            edge_label_index[:, torch.cat([train_indices, val_indices], dim=0)],
            sparse_size=self.sparse_size,
        ).sort_by('row')[0]
        # if not self.hetero:
        #     self.train_edge_weights = edge_weights[train_indices]
        #     self.val_edge_weights = edge_weights[val_indices]
        #     self.test_edge_weights = edge_weights[test_indices]
        #     self.train_val_edge_weights = edge_weights[torch.cat([train_indices, val_indices], dim=0)]
        # if self.hetero:
        self.train_edge_weights = None
        self.val_edge_weights = None
        self.test_edge_weights = None
        self.train_val_edge_weights = None
        return 
    
    def get_loaders(self, batch_size=2048, seed=None):
        if seed is None:
            seed = self.seed
        seed_everything(seed)

        if self.hetero:
            train_data = self.data.clone()

            train_data['user', 'rates', 'business'].edge_index = self.train_edge_label_index
            train_data['business', 'rev_rates', 'user'].edge_index = self.train_edge_label_index.flip(dims=(0,))
            train_data['user', 'rates', 'business'].edge_label_index = self.train_edge_label_index
            train_data['user', 'rates', 'business'].edge_label = torch.ones(self.train_edge_label_index.size(1))

            val_data = self.data.clone()
            val_data['user', 'rates', 'business'].edge_index = self.train_edge_label_index
            val_data['business', 'rev_rates', 'user'].edge_index = self.train_edge_label_index.flip(dims=(0,))

            val_data['user', 'rates', 'business'].edge_label_index = self.val_edge_label_index
            val_data['user', 'rates', 'business'].edge_label = torch.ones(self.val_edge_label_index.size(1))

            test_data = self.data.clone()
            test_data['user', 'rates', 'business'].edge_index = self.train_val_edge_label_index
            test_data['business', 'rev_rates', 'user'].edge_index = self.train_val_edge_label_index.flip(dims=(0,))
            test_data['user', 'rates', 'business'].edge_label_index = self.test_edge_label_index



            # train_loader = LinkNeighborLoader(
            #     data=train_data,
            #     num_neighbors=[-1, -1],
            #     edge_label_index=(("user", "rates", "business"), self.train_edge_label_index),
            #     neg_sampling=dict(mode='binary', amount=1),
            #     shuffle=True,
            # )
            # val_loader = LinkNeighborLoader(
            #     data=val_data,
            #     num_neighbors=[-1, -1],
            #     edge_label_index=(("user", "rates", "business"), self.val_edge_label_index),
            #     neg_sampling=dict(mode='binary', amount=1),
            #     shuffle=True,
            # )
            # src_loader = NeighborLoader(
            #     input_nodes='user',
            #     data=self.data,
            #     num_neighbors=[5, 5, 5],
            #     batch_size=batch_size*2,
            # )
            # dst_loader = NeighborLoader(
            #     input_nodes='business',
            #     data=self.data,
            #     num_neighbors=[5, 5, 5],
            #     batch_size=batch_size*2,
            # )
            return train_data, val_data, test_data #train_loader, val_loader, src_loader, dst_loader
        else:

            train_loader = torch.utils.data.DataLoader(
                range(self.train_edge_label_index.size(1)),
                shuffle=True,
                batch_size=2048,
            )

            return train_loader
        

    def split_and_loaders(self, seed=None, batch_size=128):

        self.set_data_split(seed)

        return self.get_loaders(batch_size, seed=seed)
    
    def get_test_data(self):
        return self.test_edge_label_index, self.train_val_edge_label_index, self.train_val_edge_weights
    
    def get_val_data(self):
        return self.val_edge_label_index, self.train_edge_label_index, self.train_edge_weights

    def map_postal_code(self, postal_codes):
        return self.postal_code_map[postal_codes].values

    def original_postal_code(self, mapped_postal_codes):
        return self.unique_values[mapped_postal_codes].values


def load_demographics(states_names_in_dataset, block_groups, force_reload=False):

    if f'yelp2019_demographics.csv' in os.listdir('../datasets') and not force_reload:
        full_bg_df = pd.read_csv(f'../datasets/yelp2019_demographics.csv')
        full_bg_df.bg = full_bg_df.bg.astype(str)
        full_bg_df.set_index('bg', inplace=True)
        full_bg_df['tract'] = full_bg_df['tract'].astype(str)
        full_bg_df['tract_dcid'] = full_bg_df['tract_dcid'].astype(str)
        full_bg_df['zip'] = full_bg_df['zip'].astype(str)
        full_bg_df['zip_dcid'] = full_bg_df['zip_dcid'].astype(str)
        full_bg_df['state'] = full_bg_df['state'].astype(str)

        print('Block groups demographics already downloaded')
    else:
        states = dc.get_places_in(['country/USA'], 'State')['country/USA']

        states_in_dataset = states_names_in_dataset.keys()

        # coupling state names with their respective datacommons ids
        states_names = dc.get_property_values(states, 'name')
        states_names = {key: value[0] for key, value in states_names.items() if value[0] in states_in_dataset}

        # getting all zip codes id of the states in the dataset
        blocks_in_states = dc.get_places_in(states_names.keys(), 'CensusBlockGroup')

        # setting a df with all the zips in the dataset, plus their state, state_id, and demographic information
        full_bg_df = []

        dem_var_list = ['Median_HomeValue_HousingUnit_OccupiedHousingUnit_OwnerOccupied', 'dc/e9gftzl2hm8h9',  'Median_Income_Household', 'Median_Age_Person', 'Count_Person']
        # 'dc/e9gftzl2hm8h9' is the dcid for the variable "Commute Time"

        for state in states_names.keys():
            bgs = blocks_in_states[state]
            print(state, len(bgs))
            if len(bgs) == 0:
                continue

            tracts_in_state = dc.get_places_in([state], 'CensusTract')[state]
            blocks_in_tracts = dc.get_places_in(tracts_in_state, 'CensusBlockGroup')
            # invert the dictionary
            tracts_of_bg = {bg: tract_dcid for tract_dcid, curr_bgs in blocks_in_tracts.items() for bg in curr_bgs}
            zips_in_state = dc.get_places_in([state], 'CensusZipCodeTabulationArea')[state]
            blocks_in_zips = dc.get_places_in(zips_in_state, 'CensusBlockGroup')
            # check if any block group is in multiple zip codes
            multiple_zips_bg = {}
            for zip_dcid, curr_bgs in blocks_in_zips.items():
                for bg in curr_bgs:
                    if bg in multiple_zips_bg:
                        multiple_zips_bg[bg].append(zip_dcid)
                    else:
                        multiple_zips_bg[bg] = [zip_dcid]
            # if a block group is in multiple zip codes, print a warning
            for bg, zips in multiple_zips_bg.items():
                if len(zips) > 1:
                    print(f'Warning: block group {bg} is in multiple zip codes: {zips}')

            # invert the dictionary to get the zip of each block group code
            zips_of_bg = {bg: zip_dcid for zip_dcid, curr_bgs in blocks_in_zips.items() for bg in curr_bgs}
            
            bgs_names = {bg_dcid: bg_dcid.split('/')[-1] for bg_dcid in bgs}
            for bg_dcid, value in bgs_names.items():
                if value in block_groups:
                    curr_bg_dict = {'bg': value, 'bg_dcid': bg_dcid, 'state_dcid': state, 'state': states_names[state], 'state_abbr': states_names_in_dataset[states_names[state]]}
                    if bg_dcid in tracts_of_bg:
                        curr_bg_dict['tract_dcid'] = tracts_of_bg[bg_dcid]
                        curr_bg_dict['tract'] = tracts_of_bg[bg_dcid].split('/')[-1]
                    else:
                        curr_bg_dict['tract_dcid'] = None
                        curr_bg_dict['tract'] = None
                    if bg_dcid in zips_of_bg:
                        curr_bg_dict['zip_dcid'] = zips_of_bg[bg_dcid]
                        curr_bg_dict['zip'] = zips_of_bg[bg_dcid].split('/')[-1]
                    else:
                        curr_bg_dict['zip_dcid'] = None
                        curr_bg_dict['zip'] = None
                    for dem_var_name in dem_var_list:
                        curr_bg_dict[dem_var_name] = dc.get_stat_value(bg_dcid, dem_var_name, date='2019')
                        if bg_dcid in tracts_of_bg:
                            curr_bg_dict[f'{dem_var_name}_tract'] = dc.get_stat_value(tracts_of_bg[bg_dcid], dem_var_name, date='2019')
                        else:
                            curr_bg_dict[f'{dem_var_name}_tract'] = None
                        if bg_dcid in zips_of_bg:
                            curr_bg_dict[f'{dem_var_name}_zip'] = dc.get_stat_value(zips_of_bg[bg_dcid], dem_var_name, date='2019')
                        else:
                            curr_bg_dict[f'{dem_var_name}_zip'] = None
                    

                    full_bg_df.append(curr_bg_dict)
        full_bg_df = pd.DataFrame(full_bg_df)
        full_bg_df.set_index('bg', inplace=True)
        dem_var_change = {'Median_HomeValue_HousingUnit_OccupiedHousingUnit_OwnerOccupied': 'Median_HomeValue', 'dc/e9gftzl2hm8h9':'Commute_Time',  'Median_Income_Household': 'Median_Income', 'Median_Age_Person': 'Median_Age', 'Count_Person': 'Population'}
        for dem_var in dem_var_list:
            # rename the columns of the demographic variables
            new_col_name = dem_var_change[dem_var]
            full_bg_df.rename(columns={dem_var: new_col_name, f'{dem_var}_tract': f'{new_col_name}_tract', f'{dem_var}_zip': f'{new_col_name}_zip'}, inplace=True)
        full_bg_df.to_csv('../datasets/yelp2019_demographics.csv')

    return full_bg_df