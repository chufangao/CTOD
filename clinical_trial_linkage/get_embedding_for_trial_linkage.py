import json
import glob
from logging import root
from tqdm import tqdm
import os
import pandas as pd
from multiprocessing import Process, set_start_method, get_context
from sentence_transformers import SentenceTransformer, models
from trial_linkage_utils import  create_passage
import pickle
import torch
import argparse

def sample_process(root_folder,gpu_ids, dev = False):

    process_id = int(get_context().current_process().name.split('-')[-1]) - 1
    device_id = gpu_ids[process_id % len(gpu_ids)]
    device = f'cuda:{device_id}'

    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    # Initialize the PubMedBERT model to obtain embeddings
    embeddings = models.Transformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext").to(device)
    pooling = models.Pooling(embeddings.get_word_embedding_dimension()).to(device)
    bi_encoder = SentenceTransformer(modules=[embeddings, pooling], device=device)
    bi_encoder.max_seq_length = 512

    trial_info_path = root_folder.replace('trial_embeddings','trial_info.json')
    with open(trial_info_path, 'r') as f: #  read the extracted trial info from CITI data
        trial_info = json.load(f)


    # group_list = ['Early Phase 1', 'Phase 1', 'Phase 1/Phase 2', 'Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4']
    group_list = ['early_phase1', 'phase1', 'phase1/phase2', 'phase2', 'phase2/phase3', 'phase3', 'phase4']
    print('Dividing into phases ===>')
    phase_trials = {phase: {study: trial_info[study]
                            for study in trial_info if trial_info[study]['phase'] == phase}
                    for phase in group_list}


    
    # print the count of trials in each phase
    for phase in phase_trials:
        print(phase, len(phase_trials[phase]))
    phase_trials = create_passage(phase_trials)
    
    #extract embedding for each trial and save as a pickle file
    info_list = [ 'condition_passage', 'intervention_passage','official_title','lead_sponsor','brief_summary','eligibility']
    
    for target_phase in group_list:
        print(f'Processing {target_phase} ')
        num = 0
        for study in tqdm(phase_trials[target_phase]):
            if os.path.exists(os.path.join(root_folder, study + '.pkl')) and dev:
                continue
            info_ind = 4   # to separate the short and long info
            study_info = phase_trials[target_phase][study]
            study_info_list = [study_info[info].lower() for info in info_list]
            short_info_list = study_info_list[:info_ind]
            long_info_list = study_info_list[info_ind:]
            
            embedding_dict = {}
            short_target_embedding = bi_encoder.encode(short_info_list, convert_to_tensor=True)
            embedding_dict = {f'{info}_embedding': short_target_embedding[i].detach().cpu() for i, info in enumerate(info_list[:info_ind])}
            
            
            for long_info in long_info_list:
                #break the long_info into segments with 512 length
                long_info = long_info.split()
                long_info_segments = []
                for i in range(0,len(long_info),512):
                    long_info_segments.append(' '.join(long_info[i:i+512]))
                long_target_embedding = bi_encoder.encode(long_info_segments, convert_to_tensor=True)   
                # get average embedding
                if long_target_embedding.size(0) > 1:
                    long_target_embedding = torch.mean(long_target_embedding, dim=0).unsqueeze(0)
                else:
                    long_target_embedding = long_target_embedding
                embedding_dict[f'{info_list[info_ind]}_embedding'] = long_target_embedding.detach().cpu() 
                info_ind += 1
                
            #save the embeddings
            pickle.dump(embedding_dict, open(os.path.join(root_folder, study + '.pkl'), 'wb'))
            num+=1
            if dev and num >= 100:
                break
                #check the number of files in the folder
                        

        
    
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default='', help='path to save the data')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    # parser.add_argument('--root_folder', type=str, default = None, help='Path to save the embeddings')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--gpu_ids', type=str, default= '0,1', help='List of gpu ids to use')
    args = parser.parse_args()
    
    set_start_method('spawn')
    root_folder = os.path.join(args.save_path,'clinical_trial_linkage/trial_embeddings') # < Path to save the embeddings >
    num_workers = args.num_workers # number of workers
    gpu_ids = args.gpu_ids.split(',') # list of gpu ids to use
    
    if root_folder is None:
        raise ValueError("Please provide a valid path to save the embeddings")
    
    if not os.path.exists(root_folder):
        os.makedirs(root_folder)
    
    

    processes = []
    for i in range(num_workers):
        p = Process(target=sample_process, args=(root_folder,gpu_ids,args.dev))
        p.start()
        processes.append(p)


    for p in processes:
        p.join()

    print("All Done!")

if __name__ == '__main__':
    main()

#1