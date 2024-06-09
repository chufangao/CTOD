import json
import glob
from tqdm import tqdm
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import argparse


def get_trial_linkage_weak_outcome_labels(save_path,trial_linkage_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    phase_connect = {
    'Phase 4': ['Phase 3','Phase 2/Phase 3'],
    'Phase 3': ['Phase 2','Phase 1/Phase 2'],
    'Phase 2/Phase 3': ['Early Phase 1','Phase 1', ],
    'Phase 2': ['Phase 1', 'Early Phase 1'],
    }

    # read trial info from json
    with open('./trial_info.json', 'r') as f:
        trial_info = json.load(f)
        f.close()
    


    # Separate the trials into groups based on phase and map the intervention names to generic names
    group_list = ['Early Phase 1','Phase 1','Phase 1/Phase 2','Phase 2','Phase 2/Phase 3','Phase 3','Phase 4']

    #separate trials by phase
    phase_trials = {}
    for phase in group_list:
        phase_trials[phase] = {}
        for study in trial_info:
            if trial_info[study]['phase'] == phase:
                phase_trials[phase][study] = trial_info[study]
                
    # print the count of trials in each phase
    for phase in phase_trials:
        print(phase, len(phase_trials[phase]))
        
    
    # create trial to index and index to trial mapping
    trial_to_index_map = {}
    index_to_trial_map = {}
    for phase in list(phase_connect.keys()):
        trial_to_index_map[phase] = {}
        index_to_trial_map[phase] = {}
        
        for i, trial in enumerate(phase_trials[phase]):
            trial_to_index_map[phase][trial] = i
            index_to_trial_map[phase][i] = trial
            
        i = 0
        trial_to_index_map[f'{phase} connected'] = {}
        index_to_trial_map[f'{phase} connected'] = {}
        for connect_phase in phase_connect[phase]:
            for trial in phase_trials[connect_phase]:
                trial_to_index_map[f'{phase} connected'][trial] = i
                index_to_trial_map[f'{phase} connected'][i] = trial
                i += 1
                
        # assert
        for trial in trial_to_index_map[phase]:
            assert index_to_trial_map[phase][trial_to_index_map[phase][trial]] == trial
        for trial in trial_to_index_map[f'{phase} connected']:
            assert index_to_trial_map[f'{phase} connected'][trial_to_index_map[f'{phase} connected'][trial]] == trial

    #save the trial to index mapping
    with open(f'{save_path}/trial_to_index_map.json', 'w') as f:
        json.dump(trial_to_index_map, f)
        f.close()

    #save the index to trial mapping
    with open(f'{save_path}/index_to_trial_map.json', 'w') as f:
        json.dump(index_to_trial_map, f)
        f.close()
    


    for key in list(phase_connect.keys()):
        # create adjacency matrix and edge value matrix
        adj_matrix = np.zeros((len(trial_to_index_map[key]), len(trial_to_index_map[f'{key} connected'])))
        edge_values = np.zeros((len(trial_to_index_map[key]), len(trial_to_index_map[f'{key} connected'])))
        
        assert adj_matrix.shape[0] == len(phase_trials[key])
        print(adj_matrix.shape[1], len([phase_trials[connect_phase] for connect_phase in phase_connect[key]])   )
        assert adj_matrix.shape[1] == sum([len(phase_trials[connect_phase]) for connect_phase in phase_connect[key]])
        
        
        phase = key
        key = key.replace('/', '_')
        print(f'Processing trial linkage of {key} ')
        path_to_trial_key = os.path.join(trial_linkage_path, key)
        list_of_phase_file = glob.glob(f'{path_to_trial_key}/*.json')
        

        
        print(f'Extracting adjacency matrix and edge values for {phase}')
        for file in tqdm(list_of_phase_file):
            with open(file, 'r') as f:
                data = json.load(f)
                f.close()
        
            # get main key
            target_trial = list(data.keys())[0]
            Connected_trials_list = list(data[target_trial].keys())
            
            
            for connected_trial in Connected_trials_list:
                adj_matrix[trial_to_index_map[phase][target_trial], trial_to_index_map[f'{phase} connected'][connected_trial]] = 1
                edge_values[trial_to_index_map[phase][target_trial], trial_to_index_map[f'{phase} connected'][connected_trial]] = data[target_trial][connected_trial]['cross_encoder_scores']
            
        #save the adjacency matrix and edge values as pickle
        with open(f'{save_path}/{key}_adj_matrix.pkl', 'wb') as f:
            pickle.dump(adj_matrix, f)
            f.close()
        
        with open(f'{save_path}/{key}_edge_values.pkl', 'wb') as f:
            pickle.dump(edge_values, f)
            f.close()
        
        # extract the outcome labels for the trials in Phase 4 connected using adj_matrix_phase_4 and edge_values.
        # Conditions:
        # 1) If any edge value for the connected trial is greater than 0, then the outcome label for connected trial: Success
        # 2) If all adjacent matrix values for the connected trial are 0, then the outcome label for connected trial: Failure
        # 3) If any adjacent matrix value for the connected trial is 1, then the outcome label for connected trial: Not sure

        phase_names = [ph for ph in phase_connect[phase]]
        phase_names = ' '.join(phase_names)
        phase_names = '('+phase_names.replace('/','_')+ ')'
        
        print(f'Extracting outcome labels for {phase_names}')
        trial_linkage_outcome_labels = {}
        
        for i in tqdm(range(adj_matrix.shape[1])):
            trial = index_to_trial_map[f'{phase} connected'][i]
            trial_linkage_outcome_labels[trial] = {}
            if np.sum(adj_matrix[:,i]) == 0:
                trial_linkage_outcome_labels[trial]['outcome'] = 'Failure'
            elif any(edge_values[:,i] > 0):
                trial_linkage_outcome_labels[trial]['outcome'] = 'Success'
            else:
                trial_linkage_outcome_labels[trial]['outcome'] = 'Not sure'
                
            # get additional info (use connected phase)
            for ph in phase_connect[phase]:
                if trial in phase_trials[ph]:
                    trial_linkage_outcome_labels[trial]['phase'] = phase_trials[ph][trial]['phase']
                    break
            trial_linkage_outcome_labels[trial]['connected next phase'] = []
            if  any(edge_values[:,i] > 0):
                connected_target_index = np.where(edge_values[:,i]>0)
                for index in connected_target_index[0]:
                    temp_dict = {'trial':index_to_trial_map[phase][index],'cross_encoder_score':edge_values[index,i]}
                    trial_linkage_outcome_labels[trial]['connected next phase'].append(temp_dict)
            
            trial_linkage_outcome_labels[trial]['weakly connected next phase'] = []
            if any(adj_matrix[:,i] == 1) and not any(edge_values[:,i] > 0):
                connected_target_index = np.where(adj_matrix[:,i] == 1)
                for index in connected_target_index[0]:
                    temp_dict = {'trial':index_to_trial_map[phase][index],'cross_encoder_score':edge_values[index,i]}
                    trial_linkage_outcome_labels[trial]['weakly connected next phase'].append(temp_dict)
                    
        # convert the outcome labels to a dataframe
        outcome_df = pd.DataFrame.from_dict(trial_linkage_outcome_labels,orient='index').reset_index()
        print(f'Saving outcome labels for {phase_names} of size {outcome_df.shape}')
        outcome_df.rename(columns={'index':'nctid'},inplace=True)
        outcome_df.to_csv(os.path.join(save_path,f'{phase_names}_trial_linkage_outcome_df.csv'),index=False)
        # break
    
    print('Done extracting outcome labels for trial linkage.')
    
    #combine phase 1 and early phase 1 labels
    print('Combining Phase 1 and Early Phase 1 labels')
    phase1_df1 = pd.read_csv(os.path.join(save_path,'(Phase 1 Early Phase 1)_trial_linkage_outcome_df.csv'))
    phase1_df2 = pd.read_csv(os.path.join(save_path,'(Early Phase 1 Phase 1)_trial_linkage_outcome_df.csv'))
    
    merge_dict = {}
    nct_id_list = set(list(phase1_df1['nctid'].values) + list(phase1_df2['nctid'].values))
    print(f'Number of unique trials in Phase 1 and Early Phase 1: {len(nct_id_list)}')
    
    for trial in nct_id_list:
        merge_dict[trial] = {}
        trial_1_dict = phase1_df1[phase1_df1['nctid'] == trial]
        trial_2_dict = phase1_df2[phase1_df2['nctid'] == trial]

        merge_dict[trial]['phase'] = trial_1_dict['phase'].iloc[0]
        
        if trial_1_dict['outcome'].iloc[0] == trial_2_dict['outcome'].iloc[0]:
            merge_dict[trial]['outcome'] = trial_1_dict['outcome'].iloc[0]
            merge_dict[trial]['connected next phase'] = eval(trial_1_dict['connected next phase'].iloc[0])+eval(trial_2_dict['connected next phase'].iloc[0])
            merge_dict[trial]['weakly connected next phase'] = eval(trial_1_dict['weakly connected next phase'].iloc[0])+eval(trial_2_dict['weakly connected next phase'].iloc[0])
        else:
            if trial_1_dict['outcome'].iloc[0] == 'Success' and trial_2_dict['outcome'].iloc[0] != 'Success':
                merge_dict[trial]['outcome'] = trial_1_dict['outcome'].iloc[0]
                merge_dict[trial]['connected next phase'] = eval(trial_1_dict['connected next phase'].iloc[0])
                merge_dict[trial]['weakly connected next phase'] = eval(trial_1_dict['weakly connected next phase'].iloc[0])
                
            elif trial_2_dict['outcome'].iloc[0] == 'Success' and trial_1_dict['outcome'].iloc[0] != 'Success':
                merge_dict[trial]['outcome'] = trial_2_dict['outcome'].iloc[0]
                merge_dict[trial]['connected next phase'] = eval(trial_2_dict['connected next phase'].iloc[0])
                merge_dict[trial]['weakly connected next phase'] = eval(trial_2_dict['weakly connected next phase'].iloc[0])
                
            elif trial_1_dict['outcome'].iloc[0] == 'Not sure' and trial_2_dict['outcome'].iloc[0] == 'Failure':
                merge_dict[trial]['outcome'] = trial_1_dict['outcome'].iloc[0]
                merge_dict[trial]['connected next phase'] = eval(trial_1_dict['connected next phase'].iloc[0])
                merge_dict[trial]['weakly connected next phase'] = eval(trial_1_dict['weakly connected next phase'].iloc[0])
            
            elif trial_2_dict['outcome'].iloc[0] == 'Not sure' and trial_1_dict['outcome'].iloc[0] == 'Failure':
                merge_dict[trial]['outcome'] = trial_2_dict['outcome'].iloc[0]
                merge_dict[trial]['connected next phase'] = eval(trial_2_dict['connected next phase'].iloc[0])
                merge_dict[trial]['weakly connected next phase'] = eval(trial_2_dict['weakly connected next phase'].iloc[0])
                
    merge_df = pd.DataFrame.from_dict(merge_dict, orient='index').reset_index()
    print(f'Saving merged Phase 1 and Early Phase 1 labels of size {merge_df.shape}')
    merge_df.rename(columns={'index':'nctid'},inplace=True)
    merge_df.to_csv(os.path.join(save_path,'Merged_(Early Phase 1 Phase 1)_trial_linkage_outcome_df.csv'),index=False)
    
    # delete the individual phase 1 and early phase 1 files
    os.remove(os.path.join(save_path,'(Phase 1 Early Phase 1)_trial_linkage_outcome_df.csv'))
    os.remove(os.path.join(save_path,'(Early Phase 1 Phase 1)_trial_linkage_outcome_df.csv'))
    print('Done combining Phase 1 and Early Phase 1 labels')
    
    # combine all the phase labels
    print('Combining all phase labels')
    outcome_file_paths = glob.glob(f'{save_path}/*.csv')
    print(outcome_file_paths)
    
    all_merge_df = pd.read_csv(outcome_file_paths[0])
    for file in outcome_file_paths[1:]:
        print('Reading and merging file:',file) 
        df = pd.read_csv(file)
        all_merge_df = pd.concat([all_merge_df,df],axis=0)
    
    print(f'Saving merged outcome labels of size {all_merge_df.shape}')
    all_merge_df.to_csv(os.path.join(save_path,'Merged_(ALL)_trial_linkage_outcome_df.csv'),index=False)
    
    print('Done combining all phase labels')
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_linkage_path', type=str, default=None, help='Path to the trial linkage folder containing the json files of the trial linkage')
    args = parser.parse_args()
    
    ### Parameters
    trial_linkage_path = args.trial_linkage_path # Path to the trial linkage folder containing the json files of the trial linkage
    if trial_linkage_path is None:
        raise ValueError('Please provide the path to the trial linkage folder at trial_linkage_path')
    save_path = os.path.join(trial_linkage_path,'outcome_labels') # Path to save the extracted outcome labels, a outcome_labels path in the trial linkage folder
    print(f'Extracting outcome labels for trial linkage from {trial_linkage_path} to {save_path}')
    get_trial_linkage_weak_outcome_labels(save_path,trial_linkage_path)
    

# 3