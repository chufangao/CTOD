import json
import glob
from tqdm import tqdm
import os
import pandas as pd
from PyHealth.pyhealth import data
from trial_linkage_utils import map_drug_names
from multiprocessing import Pool, cpu_count
import argparse

def map_drug_names_wrapper(args):
    study, drug_mapping = args
    nct_id, study_data = study
    return nct_id, map_drug_names(study_data, drug_mapping)


def extract_features_trial_info(data_path,save_path):

    # Read the studies.txt file --> Basic info about the trials (nct_id, official_title, phase, start_date, completion_date)
    print('Extracting basic trial info')
    file_data = pd.read_csv(os.path.join(data_path, 'studies.txt'),sep='|',parse_dates=['start_date','completion_date'])
    print(file_data.shape)
    file_data['start_date'] = file_data['start_date'].dt.strftime('%Y-%m-%d')
    file_data['completion_date'] = file_data['completion_date'].dt.strftime('%Y-%m-%d')
    # separate out these columns ['official_title','phase','start_date','completion_date']
    file_data = file_data[['nct_id','official_title','phase','start_date','completion_date','brief_title']]
    print('before removing trials with no start_date or completion_date:',file_data.shape)


    # remove rows with nan for start_date and completion_date
    file_data = file_data.dropna(subset=['start_date','completion_date'])
    print('after removing trials with no start_date or completion_date:',file_data.shape)


    group_list = ['Early Phase 1', 'Phase 1', 'Phase 1/Phase 2', 'Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4']
    # remove rows with phase not in group_list
    file_data = file_data[file_data['phase'].isin(group_list)]
    print('after removing trials with phase not in group_list:',file_data.shape)

    #if official_title is nan, replace with brief_title
    file_data['official_title'] = file_data['official_title'].fillna(file_data['brief_title'])
    print(file_data.isnull().sum())

    # drop brief_title
    file_data = file_data.drop(columns=['brief_title'])
    
    # convert file_data to dict with nct_id as key
    trial_info = file_data.set_index('nct_id').to_dict(orient='index')
    print('Number of trials: ',len(trial_info))
    # trial_info

    # Get counts of phase
    phase_counts = {}
    for study in trial_info:
        phase = trial_info[study]['phase']
        if phase not in phase_counts:
            phase_counts[phase] = 0
        phase_counts[phase] += 1
        
    #print the count of trials in each phase
    for phase in phase_counts:
        print(phase, phase_counts[phase])
        
    
    # obtain list of lead sponsors
    print('Extracting lead sponsors')
    file_path = os.path.join(data_path, 'sponsors.txt')
    file_data = open(file_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    # print(columns)

    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        if nct_id not in trial_info:
            continue
        if study[columns.index('lead_or_collaborator')]=='lead':
            if 'lead_sponsor' not in trial_info[nct_id]:
                trial_info[nct_id]['lead_sponsor'] = []
            trial_info[nct_id]['lead_sponsor'] = study[columns.index('name')]
        

    # add None for studies without lead sponsor
    for study in trial_info:
        if 'lead_sponsor' not in trial_info[study]:
            trial_info[study]['lead_sponsor'] = ''
            
    # Extract Intervention info
    print('Extracting interventions')
    file_path = os.path.join(data_path, 'interventions.txt')
    file_data = open(file_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    # print(columns)

    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        if nct_id not in trial_info:
            continue

        if 'interventions' not in trial_info[nct_id]:
            trial_info[nct_id]['interventions'] = {'intervention_type':[], 'intervention_name':[],'intervention_description':[]}
            
        trial_info[nct_id]['interventions']['intervention_type'].append(study[columns.index('intervention_type')])
        trial_info[nct_id]['interventions']['intervention_name'].append(study[columns.index('name')])
        trial_info[nct_id]['interventions']['intervention_description'].append(study[columns.index('description')])
        
    # if there are no interventions for a study, add None
    for study in trial_info:
        if 'interventions' not in trial_info[study]:
            if 'interventions' not in trial_info[study]:
                trial_info[study]['interventions'] = {'intervention_type':'', 'intervention_name':'','intervention_description':''}
            
    # Extract conditions
    print('Extracting conditions')
    file_path = os.path.join(data_path, 'conditions.txt')
    file_data = open(file_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    print(columns)

    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        if nct_id not in trial_info:
            continue

        if 'conditions' not in trial_info[nct_id]:
            trial_info[nct_id]['conditions'] = []
            
        trial_info[nct_id]['conditions'].append(study[columns.index('downcase_name')])
        
    # if there are no conditions for a study, add None
    for study in trial_info:
        if 'conditions' not in trial_info[study]:
            trial_info[study]['conditions'] = ''
            

    # Extract eligibility criteria
    print('Extracting eligibility criteria')
    file_path = os.path.join(data_path, 'eligibilities.txt')
    file_data = open(file_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    print(columns)

    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        if nct_id not in trial_info:
            continue
        
        if 'eligibility' not in trial_info[nct_id]:
            trial_info[nct_id]['eligibility'] = {}
        trial_info[nct_id]['eligibility'] = study[columns.index('criteria')]
        
    # if there are no eligibility criteria for a study, add None
    for study in trial_info:
        if 'eligibility' not in trial_info[study]:
            trial_info[study]['eligibility'] = ''
            

    #brief summary of trial
    print('Extracting brief summaries')
    file_path = os.path.join(data_path, 'brief_summaries.txt')
    file_data = open(file_path, "r").read().strip().split('\n')
    columns = file_data[0].split('|')
    print(columns)

    for study in tqdm(file_data[1:]):
        study = study.split('|')
        nct_id = study[columns.index('nct_id')]
        if nct_id not in trial_info:
            continue
        
        if 'brief_summary' not in trial_info[nct_id]:
            trial_info[nct_id]['brief_summary'] = study[columns.index('description')]
            
    # if there are no brief summaries for a study, add None
    for study in trial_info:
        if 'brief_summary' not in trial_info[study]:
            trial_info[study]['brief_summary'] = ''
            
            
            
    # mapping drug names
    print('mapping drug names ===>')
    with open(os.path.join("./drug_mapping.json"), 'r') as f:
        drug_mapping = json.load(f)
    
    
    study_items = list(trial_info.items())
    with Pool(cpu_count()) as pool:
        for study, result in tqdm(pool.imap_unordered(map_drug_names_wrapper, [(study, drug_mapping) for study in study_items]), total=len(study_items)):
            
            trial_info[study] = result
            
    print('mapping drug names done ===>')
    print('len(trial_info):',len(trial_info))
    # save trial info to json
    with open(os.path.join(save_path, 'trial_info.json'), 'w') as f:
        json.dump(trial_info, f, indent=4)
        
    print('trial_info saved ===>',os.path.join(save_path, 'trial_info.json'))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=None, help='Path to data files folder from CITI')
    args = parser.parse_args()
    
    
    
    
    data_path = args.data_path # < Path to data files folder from CITI >
    save_path = './'
    if data_path is None:
        raise ValueError('Please provide the path to the data files from CITI at data_path')
    extract_features_trial_info(data_path,save_path)
    
# 0
        