from re import match
import pandas as pd
from datetime import datetime
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
import json
import os
from multiprocessing import Pool, Manager
from datetime import timedelta
from sentence_transformers import CrossEncoder

import warnings
warnings.filterwarnings('ignore')


def get_sub_search_group_FDA_approval(approval_dict, phase_3_trials_df):
        fda_approval_date = approval_dict['Approval_Date']
        
        # get all the trial with completion date before the approval date in the phase_3_trials_df
        phase_3_trials_df['completion_date'] = pd.to_datetime(phase_3_trials_df['completion_date'])
        phase_3_trials_df['completion_date'] = phase_3_trials_df['completion_date'].dt.strftime('%Y-%m-%d')
        # similar_trials = phase_3_trials_df[phase_3_trials_df['completion_date'] < fda_approval_date]
        similar_trials = phase_3_trials_df[phase_3_trials_df['completion_date'] < (datetime.strptime(fda_approval_date, '%Y-%m-%d') - timedelta(days=60)).strftime('%Y-%m-%d')]
        # remove the trials with completion date 2 years before the approval date
        similar_trials = similar_trials[similar_trials['completion_date'] > (datetime.strptime(fda_approval_date, '%Y-%m-%d') - timedelta(days=730)).strftime('%Y-%m-%d')]
                

        return similar_trials

def match_FDA_approval_to_trials(approval_dict, similar_trials,cross_encoder):
    ingredient = approval_dict['Ingredient'].lower()
    applicant  = approval_dict['Applicant'].lower()
    trade_name = approval_dict['Trade_Name'].lower()
    fda_approval_date = pd.to_datetime(approval_dict['Approval_Date'], format="%Y-%m-%d")
    
    # Ensure each element in the columns is a string
    similar_trials['generic_name'] = similar_trials['generic_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())
    similar_trials['intervention_name'] = similar_trials['intervention_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())

    # Filter using ingredient in intervention generic name
    matched_trials = similar_trials[similar_trials['generic_name'].apply(lambda x: ingredient in x)]
    
    # Initialize the 'generic_name_score' column
    matched_trials['generic_name_score'] = 0.0
    # Get scores from cross encoder comparing generic name with ingredient
    for i in range(matched_trials.shape[0]):
        matched_trials.at[matched_trials.index[i], 'generic_name_score'] = cross_encoder.predict([(ingredient, matched_trials.iloc[i]['generic_name'])])[0]
        
    # Get top 5 trials with highest scores
    matched_trials = matched_trials.sort_values(by='generic_name_score', ascending=False).head(5)
    

    if matched_trials.empty:
        return ''
    
    # Convert completion dates once
    matched_trials['completion_date'] = pd.to_datetime(matched_trials['completion_date'], format="%Y-%m-%d", errors='coerce')
    # Calculate the difference in days directly
    matched_trials['days_diff'] = (fda_approval_date - matched_trials['completion_date']).dt.days
    
    # Filter out rows where days_diff is negative
    matched_trials = matched_trials[matched_trials['days_diff'] >= 0]
    
    if matched_trials.empty:
        return ''
    
    # Get the trial with the smallest non-negative days_diff
    matched_trials_applicant = matched_trials.loc[matched_trials['days_diff'].idxmin(), 'nctid']
    
    return matched_trials_applicant


#### this is the good function without cross encoder
# def match_FDA_approval_to_trials(approval_dict, similar_trials):
#     ingredient = approval_dict['Ingredient'].lower()
#     applicant  = approval_dict['Applicant'].lower()
#     trade_name = approval_dict['Trade_Name'].lower()
#     fda_approval_date = pd.to_datetime(approval_dict['Approval_Date'], format="%Y-%m-%d")
    
#     # Ensure each element in the columns is a string
#     similar_trials['generic_name'] = similar_trials['generic_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())
#     similar_trials['intervention_name'] = similar_trials['intervention_name'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else x.lower())

#     # Filter using ingredient in intervention generic name
#     matched_trials = similar_trials[similar_trials['generic_name'].apply(lambda x: ingredient in x)]
#     # matched_trials_2 = similar_trials[similar_trials['intervention_name'].apply(lambda x: ingredient in x)]
#     # matched_trials_3 = similar_trials[similar_trials['generic_name'].apply(lambda x: x in ingredient)]
#     # print(matched_trials.head(), matched_trials_2.head(), matched_trials_3.head())
#     # Combine the matched trials
#     # print(similar_trials.shape)
#     # print(matched_trials.shape, matched_trials_2.shape, matched_trials_3.shape)
#     # matched_trials = pd.merge(matched_trials, matched_trials_2, how='left', on='nctid')
#     # # print(matched_trials.shape)
#     # matched_trials = pd.merge(matched_trials, matched_trials_3, how='left', on='nctid')
#     # print(matched_trials.shape)
    

#     if matched_trials.empty:
#         return ''
    
#     # Convert completion dates once
#     matched_trials['completion_date'] = pd.to_datetime(matched_trials['completion_date'], format="%Y-%m-%d", errors='coerce')
#     # Calculate the difference in days directly
#     matched_trials['days_diff'] = (fda_approval_date - matched_trials['completion_date']).dt.days
    
#     # Filter out rows where days_diff is negative
#     matched_trials = matched_trials[matched_trials['days_diff'] >= 0]
    
#     if matched_trials.empty:
#         return ''
    
#     # Get the trial with the smallest non-negative days_diff
#     matched_trials_applicant = matched_trials.loc[matched_trials['days_diff'].idxmin(), 'nctid']
    
#     return matched_trials_applicant

# def match_FDA_approval_to_trials(approval_dict, similar_trials):
#     ingredient = approval_dict['Ingredient'].lower()
#     applicant  = approval_dict['Applicant'].lower()
#     trade_name = approval_dict['Trade_Name'].lower()
#     fda_approval_date = pd.to_datetime(approval_dict['Approval_Date'], format="%Y-%m-%d")
    
#     # Filter using ingredient in intervention generic name
#     matched_trials = similar_trials[similar_trials['generic_name'].apply(lambda x: any(ingredient in s.lower() for s in x))]
#     matched_trials_2 = similar_trials[similar_trials['intervention_name'].apply(lambda x: any(ingredient in s.lower() for s in x))]
#     matched_trials_3 = similar_trials[similar_trials['generic_name'].apply(lambda x: any(s.lower() in ingredient for s in x))]
    
#     # combine the matched trials
#     print(similar_trials.shape)
#     print(matched_trials.shape, matched_trials_2.shape, matched_trials_3.shape)
#     matched_trials = pd.merge(matched_trials, matched_trials_2, how='left')
#     print(matched_trials.shape)
#     matched_trials = pd.merge(matched_trials, matched_trials_3, how='left')
#     print(matched_trials.shape)
    
    
    
#     if matched_trials.empty:
#         return ''
    
#     # # calculate jaro winkler similarity for applicant and lead_sponsor
#     # matched_trials['applicant_similarity'] = matched_trials['lead_sponsor'].apply(lambda x: jaro_winkler_similarity(applicant, x.lower()))
#     # # get the trial with the highest similarity
#     # matched_trials_applicant = matched_trials.loc[matched_trials['applicant_similarity'].idxmax(), 'nctid']
    
    
#     # Convert completion dates once
#     matched_trials['completion_date'] = pd.to_datetime(matched_trials['completion_date'], format="%Y-%m-%d", errors='coerce')
#     # Calculate the difference in days directly
#     matched_trials['days_diff'] = (fda_approval_date - matched_trials['completion_date']).dt.days
    
#     # Filter out rows where days_diff is negative
#     matched_trials = matched_trials[matched_trials['days_diff'] >= 0]
    
#     if matched_trials.empty:
#         return ''
    
#     # Get the trial with the smallest non-negative days_diff
#     matched_trials_applicant = matched_trials.loc[matched_trials['days_diff'].idxmin(), 'nctid']
    
#     return matched_trials_applicant


def match_FDA_approvals_main(save_path,merged_all_pd_path,cross_encoder):
    print('Reading the FDA approvals')
    # read the FDA approvals
    orange_book = pd.read_csv('./FDA_approvals/EOBZIP_2024_04/products.txt', sep='~',parse_dates=['Approval_Date'])
    # convert 'Approved Prior to Jan 1, 1982' to Jan 1, 1982
    orange_book['Approval_Date'] = orange_book['Approval_Date'].replace('Approved Prior to Jan 1, 1982', '1982-01-01')
    orange_book['Approval_Date'] = pd.to_datetime(orange_book['Approval_Date'])
    orange_book['Approval_Date'] = orange_book['Approval_Date'].dt.strftime('%Y-%m-%d')
    # extract only the following columns Ingredient, Trade_Name, Applicant and Approval_Date
    orange_book = orange_book[['Ingredient', 'Trade_Name', 'Applicant', 'Approval_Date']]
    # remove duplicates
    orange_book = orange_book.drop_duplicates()
    orange_book.reset_index(drop=True, inplace=True)
    print(f'Number of unique FDA approvals: {orange_book.shape[0]}')
    
    #  read trial info 
    with open('./trial_info.json', 'r') as f:
        trial_info = json.load(f)
        f.close()
    trial_info[list(trial_info.keys())[0]]

    #extract phase 3 and phase 2/ phase 3 trials with drug intervention type
    phase_3_trials = {}
    for study in trial_info:
        if trial_info[study]['phase'] == 'Phase 3' or trial_info[study]['phase'] == 'Phase 2/Phase 3':
            if 'Drug' in trial_info[study]['interventions']['intervention_type']:
                phase_3_trials[study] = trial_info[study]
    
    
    # expand the nested dictionary to a flat dictionary
    phase_3_trials_flat = {}    
    for study in phase_3_trials:
        phase_3_trials_flat[study] = {}
        for key in phase_3_trials[study]:
            if isinstance(phase_3_trials[study][key], dict):
                for k in phase_3_trials[study][key]:
                    phase_3_trials_flat[study][k] = phase_3_trials[study][key][k]
            else:
                phase_3_trials_flat[study][key] = phase_3_trials[study][key]  
                
    # convert phase_3_trials to a dataframe
    phase_3_trials_df = pd.DataFrame.from_dict(phase_3_trials_flat, orient='index')
    #rename the index to nctid
    phase_3_trials_df.reset_index(inplace=True)
    phase_3_trials_df.rename(columns={'index':'nctid'}, inplace=True)

    phase_3_trials_df['generic_name'] = phase_3_trials_df['generic_name'].apply(lambda x: list(OrderedDict.fromkeys(x)))
            
             
                
    phase_3_fda_matched = {}

    for i in tqdm(range(orange_book.shape[0])):
        approval_dict = orange_book.loc[i]
        similar_trials = get_sub_search_group_FDA_approval(approval_dict, phase_3_trials_df)
        matched_trial = match_FDA_approval_to_trials(approval_dict, similar_trials,cross_encoder)
        # print(matched_trial)
        # print(matched_trial)
        if matched_trial != '':
            phase_3_fda_matched[matched_trial] = {}
            phase_3_fda_matched[matched_trial]['Approval_Date'] = approval_dict['Approval_Date']
            phase_3_fda_matched[matched_trial]['Ingredient'] = approval_dict['Ingredient']
            phase_3_fda_matched[matched_trial]['Trade_Name'] = approval_dict['Trade_Name']
            phase_3_fda_matched[matched_trial]['Applicant'] = approval_dict['Applicant']
            # add following information intervention name, intervention generic name, start_date, completion date condition and lead sponsor
            trial_phase_3_info = phase_3_trials_df[phase_3_trials_df['nctid'] == matched_trial]
            phase_3_fda_matched[matched_trial]['intervention_name'] = trial_phase_3_info['intervention_name'].values[0]
            phase_3_fda_matched[matched_trial]['generic_name'] = trial_phase_3_info['generic_name'].values[0]
            phase_3_fda_matched[matched_trial]['start_date'] = trial_phase_3_info['start_date'].values[0]
            phase_3_fda_matched[matched_trial]['completion_date'] = trial_phase_3_info['completion_date'].values[0]
            phase_3_fda_matched[matched_trial]['condition'] = trial_phase_3_info['conditions'].values[0]
            phase_3_fda_matched[matched_trial]['lead_sponsor'] = trial_phase_3_info['lead_sponsor'].values[0]
  
    print(f'Number of matched trials: {len(phase_3_fda_matched)}')
    
    
    # convert to dataframe
    phase_3_fda_matched_df = pd.DataFrame.from_dict(phase_3_fda_matched, orient='index')
    phase_3_fda_matched_df.reset_index(inplace=True)
    phase_3_fda_matched_df.rename(columns={'index':'nctid'},inplace=True)
    phase_3_fda_matched_df.to_csv(os.path.join(save_path, 'phase_3_fda_matched.csv'), index=False)
    
    
    print('Finished matching FDA approvals to trials')

    
    # read the merged_all_pd
    merged_all_pd = pd.read_csv(merged_all_pd_path)
    
    # check if nctid is in phase_3_fda_matched
    # if true and outcome in merged_all_pd is not Success convert to Success
    # else leave it
    
    for i in range(merged_all_pd.shape[0]):
        nctid = merged_all_pd.iloc[i]['nctid']
        if nctid in phase_3_fda_matched_df['nctid'].values:
            if merged_all_pd.iloc[i]['outcome'] != 'Success':
                merged_all_pd.at[i, 'outcome'] = 'Success'

    merge_all_save_path = merged_all_pd_path.split('.csv')[0] + '_FDA_updated.csv'
    merged_all_pd.to_csv(merge_all_save_path, index=False)
    
    print('Finished updating merged_all_pd with FDA approvals')
    
if __name__ == '__main__':
    save_path = '/srv/local/data/jp65/trial_linkage_5_feat_(no_eligibility)_wei_2_2_1_1_half'
    # /srv/local/data/jp65/trial_linkage_(official_title_intervention_brief_summary_eligibility_condition)/outcome_labels/Merged_(ALL)_trial_linkage_outcome_df.csv
    merged_all_pd_path = '/srv/local/data/jp65/trial_linkage_5_feat_(no_eligibility)_wei_2_2_1_1_half/outcome_labels/Merged_(ALL)_trial_linkage_outcome_df.csv'
    # merged_all_pd_path = '/srv/local/data/jp65/trial_linkage_(official_title_intervention_brief_summary_eligibility_condition)/outcome_labels/Merged_(ALL)_trial_linkage_outcome_df.csv'
    device = 'cuda'
    cross_encoder = CrossEncoder(model_name='cross-encoder/ms-marco-MiniLM-L-12-v2', device=device)

    
    print(f'Matching FDA approvals to trials and updating the merged_all_pd at {merged_all_pd_path}')
    match_FDA_approvals_main(save_path,merged_all_pd_path,cross_encoder)
    
# conda activate surv_llm
# cd /home/jp65/CTOD/trial_linkage_main_codes/main_revised
#python match_fda_approvals_4.py