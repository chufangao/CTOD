import json
import glob
from logging import root
from networkx import intersection
from tqdm import tqdm
import os
import pandas as pd
from multiprocessing import Process, set_start_method, get_context, Queue, Manager
from sentence_transformers import SentenceTransformer, util, models, CrossEncoder
from trial_linkage_utils import create_passage, create_connected_phase_dict,get_top_k_ancestors
import pickle
import torch
from datetime import datetime
from collections import OrderedDict

def get_sub_search_group(target_trial_dict, phase_trials):
        datetime_format = "%Y-%m-%d"
        target_start_date = target_trial_dict['start_date']
        target_start_date = datetime.strptime(target_start_date, datetime_format)
        
        
        # print(target_start_date)
        target_intervention_type = list(OrderedDict.fromkeys(target_trial_dict['interventions']['intervention_type']))
        target_intervention_generic = list(OrderedDict.fromkeys(target_trial_dict['interventions']['generic_name']))
        # target_condition = list(OrderedDict.fromkeys(target_trial_dict['conditions']))
        
        similar_trials = {}
        for trial in phase_trials:
            trial_dict = phase_trials[trial]
            
            if trial_dict['completion_date'] == '':
                continue
            trial_completion_date = trial_dict['completion_date']
            trial_completion_date = datetime.strptime(trial_completion_date, datetime_format)
            
            if trial_completion_date <= target_start_date:
                #get intersection of intervention types
                trial_intervention_type = list(OrderedDict.fromkeys(trial_dict['interventions']['intervention_type']))
                intersection_intervention_type = list(set(target_intervention_type).intersection(trial_intervention_type))
       
                if len(intersection_intervention_type) > 0:
                    if 'Drug' in target_intervention_type:
                        trial_intervention_generic = list(OrderedDict.fromkeys(trial_dict['interventions']['generic_name']))
                        # check if any generic name in target_intervention_generic has a overlap with any string in trial_intervention_generic and vice versa
                        if any([any([drug in trial_drug for trial_drug in trial_intervention_generic]) for drug in target_intervention_generic]):
                            similar_trials[trial] = trial_dict
                        elif any([any([drug in target_drug for target_drug in target_intervention_generic]) for drug in trial_intervention_generic]):
                            similar_trials[trial] = trial_dict
                    else:
                        similar_trials[trial] = trial_dict
        return similar_trials

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

    with open('/home/jp65/CTOD/trial_linkage_main_codes/main_revised/trial_info.json', 'r') as f:
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
    
    # for study in tqdm(phase_trials[target_phase]):
        # try:
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
            # break
        # except Exception as e:
        #     print(f'Error in {study} : {e}')
        #     progress_dict[study] = 'done'
        #     continue
        
def main():
    set_start_method('spawn', force=True)
    # data_path = '/home/jp65/CTOD/data'
    # root_folder = '/srv/local/data/jp65/trial_linkage_5_feat_(no_eligibility)_wei_2_2_1_1_half'
    root_folder = '/srv/local/data/jp65/trial_linkage_(official_title2_intervention2_brief_summary1_eligibility1_condition2_lead_sponsor_(only_4,2))'
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    target_phase = 'Phase 2/Phase 3' #'Phase 2/Phase 3' #'Phase 2'
    embedding_path = '/srv/local/data/jp65/trial_linkage_6_feat_embeddings_revised'

    # features to use for linking
    
    ## main label : best for now
    info_list = [ 'official_title','intervention_passage','brief_summary','eligibility','condition_passage']#,'lead_sponsor']#,'eligibility']
    info_wei_list = {'official_title': 2,'intervention_passage': 2,'brief_summary':1,'eligibility':1,'condition_passage':2}#,'lead_sponsor':2}
    ##
    
    # info_list = [ 'lead_sponsor']#, 'intervention_passage','official_title','lead_sponsor','brief_summary']#,'eligibility']
    # info_wei_list = {'lead_sponsor': 1}#, 'intervention_passage': 2, 'official_title': 1, 'lead_sponsor': 1, 'brief_summary': 0.5}

    print(f'Creating linkage for {target_phase}')
    print(f'Using features: {info_list}')
    print(f'Using feature weights: {info_wei_list}')
    # ['Early Phase 1', 'Phase 1', 'Phase 1/Phase 2', 'Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4']
    num_workers = 1
    gpu_ids = [7]
    
    with open('/home/jp65/CTOD/trial_linkage_main_codes/main_revised/trial_info.json', 'r') as f:
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


# conda activate surv_llm
#cd /home/jp65/CTOD/trial_linkage_main_codes/main_revised/

#python create_trial_linkage_2_revised_efficient_new_idea_test.py