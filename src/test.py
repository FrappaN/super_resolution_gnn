import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"

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




def load_demographics(states_names_in_dataset, block_groups):

    states = dc.get_places_in(['country/USA'], 'State')['country/USA']

    states_in_dataset = states_names_in_dataset.keys()

    # coupling state names with their respective datacommons ids
    states_names = dc.get_property_values(states, 'name')
    states_names = {key: value[0] for key, value in states_names.items() if value[0] in states_in_dataset}

    # getting all zip codes id of the states in the dataset
    blocks_in_states = dc.get_places_in(states_names.keys(), 'CensusBlockGroup')

    # setting a df with all the zips in the dataset, plus their state, state_id, and demographic information
    full_bg_df = []

    dem_var_list = ['Median_HomeValue_HousingUnit_OccupiedHousingUnit_OwnerOccupied', 'dc/e9gftzl2hm8h9',  'Median_Income_Household', 'Median_Age_Person']
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
    dem_var_change = {'Median_HomeValue_HousingUnit_OccupiedHousingUnit_OwnerOccupied': 'Median_HomeValue', 'dc/e9gftzl2hm8h9':'Commute_Time',  'Median_Income_Household': 'Median_Income', 'Median_Age_Person': 'Median_Age'}
    for dem_var in dem_var_list:
        # rename the columns of the demographic variables
        new_col_name = dem_var_change[dem_var]
        full_bg_df.rename(columns={dem_var: new_col_name, f'{dem_var}_tract': f'{new_col_name}_tract', f'{dem_var}_zip': f'{new_col_name}_zip'}, inplace=True)
    #full_bg_df.to_csv('../datasets/yelp2019_demographics.csv')

    return full_bg_df


if __name__ == "__main__":
    business_df = pd.read_csv(f'../datasets/yelp2019_business.csv')

    business_df.loc[:, 'postal_code'] = business_df['postal_code'].fillna(0)
    business_df['postal_code'] = business_df['postal_code'].astype(str)
    business_df.loc[:, 'block_group_id'] = business_df['block_group_id'].fillna(0)
    business_df['block_group_id'] = business_df['block_group_id'].astype(int).astype(str)

    block_groups = business_df['block_group_id'].unique()


    # converting full names into abbreviations
    states_names ={
        'Missouri': 'MO',
        'Pennsylvania': 'PA',
        'Tennessee': 'TN',
        'Florida': 'FL',
        'Indiana': 'IN',
        }
    demographics_df = load_demographics(states_names, block_groups)