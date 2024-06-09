import json
import glob
from logging import root
from networkx import intersection
from tqdm import tqdm
import os
import pandas as pd
from multiprocessing import Process, set_start_method, get_context, Queue, Manager
from sentence_transformers import SentenceTransformer, util, models, CrossEncoder
from trial_linkage_utils import create_passage, create_connected_phase_dict,get_top_k_ancestors, get_sub_search_group
import pickle
import torch
from datetime import datetime
from collections import OrderedDict
import argparse



def get_trial_linkage(root_folder,embedding_path,target_phase,info_list, info_wei_list,gpu_ids,task_queue,progress_dict):
    set_start_method('spawn', force=True)
    
    if target_phase == 'Phase 1/Phase 2':
        save_target_phase = 'Phase 1_Phase 2'
    elif target_phase == 'Phase 2/Phase 3':
        save_target_phase = 'Phase 2_Phase 3'
    else:
        save_target_phase = target_phase
    
    
    process_id = int(get_context().current_process().name.split('-')[-1]) - 1
    device_id = gpu_ids[process_id % len(gpu_ids)]
    device = f'cuda:{device_id}'

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
        
    if not os.path.exists(os.path.join(root_folder,save_target_phase)):
        os.makedirs(os.path.join(root_folder,save_target_phase))
    

    cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)

    with open('./trial_info.json', 'r') as f:
        trial_info = json.load(f)

    # 1) Separate the trials into groups based on phase and map the intervention names to generic names

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
    
    
    # 2) Create a passage for each trial for intervention and conditions in the phase considering following:
    phase_trials = create_passage(phase_trials)
    
    
    # Dictionary of connected phases
    phase_connect = {
        'Phase 4': ['Phase 3','Phase 2/Phase 3'],
        'Phase 3': ['Phase 2','Phase 1/Phase 2'],
        'Phase 2/Phase 3': ['Phase 1', 'Early Phase 1'],
        'Phase 2': ['Phase 1', 'Early Phase 1'],
        
    }
    
    ## create search space based on the relation between phases
    connected_phase_search_space = create_connected_phase_dict(phase_trials, phase_connect)  
    
    # read all the embeddings for the trials in connected_phase_search_space[target_phase]
    embeddings_dict = {} 
    for sub_study in connected_phase_search_space[target_phase]:
        # for sub_study in connected_phase_search_space[target_phase][ph]:
            with open(os.path.join(embedding_path,sub_study + '.pkl'), 'rb') as f:
                embeddings_dict[sub_study] = pickle.load(f)
    
    # combine the embeddings
    combined_embeddings = {}
    for sub_study in embeddings_dict:
        sub_study_embedding_list = []
        sub_study_embedding = embeddings_dict[sub_study]
        for info in info_list:
            if sub_study_embedding[f'{info}_embedding'].shape[0] == 0:
                sub_study_embedding_list.append(torch.zeros(1,768))
            else:
                sub_study_embedding_list.append(sub_study_embedding[f'{info}_embedding'].squeeze().unsqueeze(0))
        combined_embeddings[sub_study] = torch.cat(sub_study_embedding_list, dim=0)


    # 3) Extract possible ancestors for trials in target phase in the connected search space
    print(f'Extracting trial linkage for trials in {target_phase}')
    
    
    
    while not task_queue.empty():
        study = task_queue.get()
    
        if os.path.exists(os.path.join(root_folder, save_target_phase, f'{study}.json')):
            progress_dict[study] = 'done'
            continue
        
        possible_ancestors = {}
        target_trial_dict = phase_trials[target_phase][study]
        
        sub_search_group = get_sub_search_group(target_trial_dict, connected_phase_search_space[target_phase])
        if len(sub_search_group) == 0:
            print(f'No search group found for {study}')
            progress_dict[study] = 'done'
            continue
        top_k_ancestors = get_top_k_ancestors(target_trial_dict,
                                                study, 
                                                sub_search_group,
                                                embedding_path,
                                                combined_embeddings,
                                                info_list,
                                                info_wei_list,
                                                cross_encoder,
                                                device, 
                                                top_k=5)
        possible_ancestors[study] = top_k_ancestors
        
        # save the possible ancestors
        with open(os.path.join(root_folder,save_target_phase,study + '.json'), 'w') as f:
            json.dump(possible_ancestors,f)
        progress_dict[study] = 'done'
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', type=str, default = None, help='Path to save the linkages')
    parser.add_argument('--target_phase', type=str, default = 'Phase 2/Phase 3', help='Phase to create linkage with the previous phases. select from ["Phase 2", "Phase 2/Phase 3", "Phase 3", "Phase 4"]')
    parser.add_argument('--embedding_path', type=str, default = None, help='Path to the embeddings folder')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--gpu_ids', type=str, default= '0,1', help='List of gpu ids to use')
    args = parser.parse_args()
    
    set_start_method('spawn', force=True)
    root_folder = args.root_folder  # < Folder to save the created linkages >
    target_phase = args.target_phase # Phase to create linkage with the previous phases. select from ['Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4']
    embedding_path = args.embedding_path # < Folder containing the embeddings saved for the trials >
    num_workers = args.num_workers # number of workers
    gpu_ids = args.gpu_ids.split(',') # list of gpu ids to use
    
    # features to use for linking 
    info_list = [ 'official_title','intervention_passage','brief_summary','eligibility','condition_passage']
    info_wei_list = {'official_title': 1,'intervention_passage': 1,'brief_summary':1,'eligibility':1,'condition_passage':1}
    
    
    print(f'Creating linkage for {target_phase}')
    print(f'Using features: {info_list}')
    print(f'Using feature weights: {info_wei_list}')
    
    
    
    if root_folder is None:
        raise ValueError('Please provide a folder to save the linkages at root_folder')
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    if embedding_path is None:
        raise ValueError('Please provide the folder containing the embeddings at embedding_path')
    
    with open('./trial_info.json', 'r') as f:
        trial_info = json.load(f)

    task_queue = Queue()
    manager = Manager()
    progress_dict = manager.dict()
    for study in trial_info:
        if trial_info[study]['phase'] == target_phase:
            task_queue.put(study)
            progress_dict[study] = 'pending'

    processes = []
    for _ in range(num_workers):
        p = Process(target=get_trial_linkage, args=(root_folder, embedding_path, target_phase, info_list, info_wei_list, gpu_ids, task_queue, progress_dict))
        p.start()
        processes.append(p)

    # Monitoring progress
    with tqdm(total=len(progress_dict)) as pbar:
        while any(p.is_alive() for p in processes):
            done_count = list(progress_dict.values()).count('done')
            pbar.n = done_count
            pbar.refresh()
        
    for p in processes:
        p.join()

    print("All Done!")

if __name__ == '__main__':
    main()


#2