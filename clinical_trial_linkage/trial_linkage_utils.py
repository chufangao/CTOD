
from datetime import datetime
import re
from tqdm import tqdm
from sentence_transformers import util
import torch
import pickle
import os 
from collections import OrderedDict



# map drug names to generic names if the intervention is a drug
def map_drug_names(target_phase_trial_dict,drug_mapping):
    generic_name_list = []
    # get the set of drugs in target_phase_trial_dict['interventions']['intervention_name']
    if target_phase_trial_dict['interventions']['intervention_name'] == '':
        target_phase_trial_dict['interventions']['generic_name'] = generic_name_list
        return target_phase_trial_dict
    elif 'Drug' not in target_phase_trial_dict['interventions']['intervention_type']:
        generic_name_list = target_phase_trial_dict['interventions']['intervention_name']
        for i in range(len(generic_name_list)):
            generic_name_list[i] = clean_string(generic_name_list[i]).lower()
        target_phase_trial_dict['interventions']['generic_name'] = generic_name_list
        return target_phase_trial_dict
    
   
    drug_name_list = list(OrderedDict.fromkeys(target_phase_trial_dict['interventions']['intervention_name']))

    for i in range(len(drug_name_list)):
        # do mapping only if the intervention is a drug
        if target_phase_trial_dict['interventions']['intervention_type'][i] == 'Drug':
            
            drug_name = clean_string(drug_name_list[i]).lower()
            if drug_name == 'placebo':
                generic_name_list.append('placebo')
                continue
            
            flag = 0
            max_score = 0
            max_score_drug = drug_name
            for drug in drug_mapping:
                if clean_string(drug).lower() in drug_name and len(clean_string(drug).lower()) > max_score:
                    max_score = len(clean_string(drug).lower())
                    max_score_drug = drug
                    flag = 1
            if flag == 1:
                generic_name_list.append(', '.join(drug_mapping[max_score_drug]))
                
            else:
                generic_name_list.append(drug_name)
        else:
            generic_name_list.append(clean_string(drug_name_list[i]).lower())
    
            
        
    target_phase_trial_dict['interventions']['generic_name'] = generic_name_list
    return target_phase_trial_dict

def get_sub_search_group(target_trial_dict, phase_trials):
        datetime_format = "%Y-%m-%d"
        target_start_date = target_trial_dict['start_date']
        target_start_date = datetime.strptime(target_start_date, datetime_format)
        
        
        target_intervention_type = list(OrderedDict.fromkeys(target_trial_dict['interventions']['intervention_type']))
        target_intervention_generic = list(OrderedDict.fromkeys(target_trial_dict['interventions']['generic_name']))
        
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
    

def remove_superscripts(text):
    # Define a regular expression pattern to match superscripted characters
    pattern = r"[\u00B2\u00B3\u00B9\u2070\u2074\u2075\u2076\u2077\u2078\u2079\u207A\u207B\u207C\u207D\u207E\u207F]"
    
    # Use the re.sub() function to replace the matched superscripts with an empty string
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text

def remove_special_chars(text):
    # Define a regular expression pattern to match superscripted characters and registered trademark symbol
    pattern = r"[\u00B2\u00B3\u00B9\u2070\u2074\u2075\u2076\u2077\u2078\u2079\u207A\u207B\u207C\u207D\u207E\u207F\u00AE]"
    
    # Use the re.sub() function to replace the matched characters with an empty string
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text

def clean_string(text):
    # Remove superscripts
    text = remove_superscripts(text)
    
    # Remove special characters
    text = remove_special_chars(text)
    
    return text


def create_passage(phase_trials):
    for phase in tqdm(phase_trials):
        for study in phase_trials[phase]:
            
            #Intervention String
            drug_name_list = list(OrderedDict.fromkeys(phase_trials[phase][study]['interventions']['generic_name']))
            phase_trials[phase][study]['intervention_passage'] = f'''{', '.join(drug_name_list)}'''
            # Condition String
            condition_list = list(OrderedDict.fromkeys(phase_trials[phase][study]['conditions']))
            phase_trials[phase][study]['condition_passage'] = f'''{', '.join(condition_list)}'''
            
    return phase_trials

def create_connected_phase_dict(phase_trials, phase_connect):
    combined_search_space = {}
    for phase in tqdm(phase_trials):
        if phase in combined_search_space:
            continue
        if phase in phase_connect:
            connected_phases = phase_connect[phase]
            # print(connected_phases)
            combined_search_space[phase] = {}
            for connected_phase in connected_phases:
                combined_search_space[phase].update(phase_trials[connected_phase])
    return combined_search_space




def get_top_k_ancestors(target_trial_dict,study, sub_search_trials,embedding_path,combined_embeddings,info_list,info_wei_list,cross_encoder,device, top_k=5):

    
    # information to be used for similarity score and weights
    wei_list =  [info_wei_list[info] for info in info_list]
    wei_list = [x/sum(wei_list) for x in wei_list]
    # convert the wei list to a diagonal matrix
    wei_list = torch.tensor(wei_list).to(device).unsqueeze(0)
    # obtain the target embeddings
    target_info_list =  [target_trial_dict[info].lower() for info in info_list]
    target_embeddings = pickle.load(open(os.path.join(embedding_path, study + '.pkl'), 'rb'))
    # concat the target embeddings to tensor
    embed_list = []
    for info in info_list:
        if target_embeddings[f'{info}_embedding'].shape[0] == 0:
            embed_list.append(torch.zeros(1,768))
        else:
            embed_list.append(target_embeddings[f'{info}_embedding'].squeeze().unsqueeze(0))
    
    
    target_embeddings = torch.cat(embed_list, dim=0)
    
    # create batches of sub_search_trials 
     #2048
    sub_search_trials_list = list(sub_search_trials.keys())
    if len(sub_search_trials_list) > 30000:
        batch_size = 30000
    else:
        batch_size = len(sub_search_trials_list)
    sub_search_trials_list = [sub_search_trials_list[i:i + batch_size] for i in range(0, len(sub_search_trials_list), batch_size)]
    
    for sub_search_trials_batch in sub_search_trials_list:
        sub_search_embedding_list = []
        for trial in sub_search_trials_batch:
            # read from the embeddings_dict
            sub_search_embedding_list.append(combined_embeddings[trial]) 
            
        sub_search_embeddings = torch.cat(sub_search_embedding_list, dim=0)
        # print(f'sub search embeddings shape: {sub_search_embeddings.shape}')
        cosine_scores = util.cos_sim(target_embeddings, sub_search_embeddings).to(device)
        # print(f'cosine scores shape: {cosine_scores.shape}')
        for i, trial in enumerate(sub_search_trials_batch):
            sub_search_trials[trial]['similarity_score'] = (wei_list @ torch.diagonal(cosine_scores[:,i*len(info_list):(i+1)*len(info_list)]).unsqueeze(1)).item()
    
    # Sort the trials based on similarity score
    top_k_trials = dict(sorted(sub_search_trials.items(), key=lambda x: x[1]['similarity_score'], reverse=True)[:32])
    
    # rerank the top 32 trials using cross encoder
    top_k_trials_list = list(top_k_trials.keys())
    
    
    cross_encoder_input = []
    for trial in top_k_trials_list:
        sub_search_info_list = [sub_search_trials[trial][info] for info in info_list]
        cross_encoder_input.extend([[target_info, sub_search_info] for target_info, sub_search_info in zip(target_info_list, sub_search_info_list)])
    
    cross_encoder_scores = cross_encoder.predict(cross_encoder_input)
    cross_encoder_scores = torch.tensor(cross_encoder_scores).reshape(-1,len(info_list))
    cross_encoder_scores = cross_encoder_scores.to(device) @ wei_list.T
    cross_encoder_scores = cross_encoder_scores.reshape(-1)
    
    for i, trial in enumerate(top_k_trials_list):
        top_k_trials[trial]['cross_encoder_score'] = cross_encoder_scores[i].item()
    
    
    # Sort the trials based on cross encoder score and select top 5
    top_k_trials = dict(sorted(top_k_trials.items(), key=lambda x: x[1]['cross_encoder_score'], reverse=True)[:top_k])
    
    # # if all the top_k trials have cross encoder score less than 0, then return all the trials
    # else:
    if all([top_k_trials[trial]['cross_encoder_score'] <= 0 for trial in top_k_trials]):
        top_k_trials = {trial: top_k_trials[trial] for trial in top_k_trials}
    else:
        # remove the trials with cross encoder score less than 0
        top_k_trials = {trial: top_k_trials[trial] for trial in top_k_trials if top_k_trials[trial]['cross_encoder_score'] > 0}
            
    
     # extract only the nct_id, similarity score, cross encoder score
    top_k_trials = {trial: {'nct_id': trial,'similarity_score': sub_search_trials[trial]['similarity_score'], 'cross_encoder_scores': sub_search_trials[trial]['cross_encoder_score']} for trial in top_k_trials}
    
    return top_k_trials