import re
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
from pyhealth.metrics import binary_metrics_fn
import glob
import json
from sklearn.metrics import cohen_kappa_score
import os
import argparse



def eval_TOP(test_pd, TOP_path,phase,test_only = True):
    '''
    phase: str, phase of the trials (phase_I, phase_II, phase_III, all) 
    '''
    
    # map # Successs and Not sure --> 1 and Failure --> 0
    test_pd = test_pd.dropna(subset=['outcome'])
    test_pd['outcome'] = test_pd['outcome'].apply(lambda x: 1 if x in ['Success']  else 0) #'Success','Not sure'
    
    print('Reading TOP dataset')
    if phase == 'all':
        if test_only:
            top_data_list = []
            for file in os.listdir(TOP_path):
                if file.endswith('test.csv'):
                    print(f'Reading file: {os.path.join(TOP_path, file)}')
                    top_data_list.append(pd.read_csv(os.path.join(TOP_path, file)))

            TOP_pd = pd.concat(top_data_list).reset_index(drop=True)
            TOP_pd.drop_duplicates(inplace=True)
        else:
            top_data_list = []
            for file in os.listdir(TOP_path):
                if file.endswith('.csv'):
                    print(f'Reading file: {os.path.join(TOP_path, file)}')
                    top_data_list.append(pd.read_csv(os.path.join(TOP_path, file)))

            TOP_pd = pd.concat(top_data_list).reset_index(drop=True)
            TOP_pd.drop_duplicates(inplace=True)
    else:
        if test_only:
            print(f'Reading {os.path.join(TOP_path,f"{phase}_test.csv")}')
            TOP_pd = pd.read_csv(os.path.join(TOP_path,f'{phase}_test.csv'))
        else:
            print(f'Reading {os.path.join(TOP_path,f"{phase}_train.csv")}')
            TOP_train = pd.read_csv(os.path.join(TOP_path,f'{phase}_train.csv'))
            print(f'Reading {os.path.join(TOP_path,f"{phase}_valid.csv")}') 
            TOP_val = pd.read_csv(os.path.join(TOP_path,f'{phase}_valid.csv'))
            print(f'Reading {os.path.join(TOP_path,f"{phase}_test.csv")}')
            TOP_test = pd.read_csv(os.path.join(TOP_path,f'{phase}_test.csv'))
            
            TOP_pd = pd.concat([TOP_train,TOP_val,TOP_test])
            
    
    # drop duplicates in TOP_pd
    TOP_pd.drop_duplicates(inplace=True)
    
    # get common nctids
    common_nctids = set(test_pd['nctid'].values).intersection(set(TOP_pd['nctid'].values))
    print(f'Number of common nctids: {len(common_nctids)}')
    
    # get the common nctids from both dataframes
    test_pd = test_pd[test_pd['nctid'].isin(common_nctids)]
    TOP_pd = TOP_pd[TOP_pd['nctid'].isin(common_nctids)]
    
    print(f'Length of test_pd: {test_pd.shape[0]}')
    print(f'Length of TOP_pd: {TOP_pd.shape[0]},Number of positive labels: {TOP_pd["label"].sum()}')
    
    #sort
    test_pd = test_pd.sort_values(by = 'nctid')
    test_pd = test_pd.reset_index(drop=True)
    TOP_pd = TOP_pd.sort_values(by = 'nctid')
    TOP_pd = TOP_pd.reset_index(drop=True)
    
    # assert whether both pdfs have same nctids in same order
    assert all(test_pd['nctid'].values == TOP_pd['nctid'].values)
    
    # get the outcome labels
    linkage_outcome = test_pd['outcome'].values
    TOP_pd_outcome = TOP_pd['label'].values
    
    # calculate the metrics (#y true , # y pred)
    results = binary_metrics_fn(TOP_pd_outcome,linkage_outcome, 
                                metrics=['accuracy', 'precision', 'recall','f1'])
    results['Number of trials'] = len(linkage_outcome)
    # Cohen's Kappa
    results['Cohen Kappa'] = cohen_kappa_score(TOP_pd_outcome,linkage_outcome)
    
    #print results:
    print(f'Accuracy: {results["accuracy"]}')
    print(f'Precision: {results["precision"]}')
    print(f'Recall: {results["recall"]}')
    print(f'F1: {results["f1"]}')
    print(f'Cohen Kappa: {results["Cohen Kappa"]}')
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trial_outcome_path', type=str, default = None, help='Path to the trial linkage outcomes saved')
    parser.add_argument('--top_data_path', type=str, default = None, help='Path to the TOP dataset saved')
    args = parser.parse_args()
    ## Set the paths
    trial_outcome_path = args.trial_outcome_path # < Path to the trial linkage outcomes saved >
    top_data_path = args.top_data_path # < Path to the TOP dataset saved >
    
    
    if trial_outcome_path is None:
        raise ValueError('Please provide the path to the trial linkage outcomes saved')
    linkage_pd_path =  os.path.join(trial_outcome_path,'outcome_labels','Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv')
    if top_data_path is None:
        raise ValueError('Please provide the path to the TOP dataset saved')
    
    save_path = f'./results/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    linkage_pd = pd.read_csv(linkage_pd_path)
    
    ### Results for trial linkage outcomes test split
    # for test set linkage pd phase 1
    result1 = eval_TOP(linkage_pd, top_data_path, 'phase_I', test_only = True)
    # for test set linkage pd phase2
    result2 = eval_TOP(linkage_pd, top_data_path, 'phase_II', test_only = True)
    # for test set linkage pd phase3
    result3 = eval_TOP(linkage_pd, top_data_path, 'phase_III', test_only = True)
    # for test set linkage pd all
    result4 = eval_TOP(linkage_pd, top_data_path, 'all', test_only = True)

    # combine all the results and save it as a csv
    results = pd.DataFrame([result1,result2,result3,result4])
    # round all the values to .4f
    results = results.round(4)
    results['phase'] = ['phase_I','phase_II','phase_III','all']
    results.to_csv(os.path.join(save_path,'trial_linkage_TOP_test_split_results.csv'),index=False)
    
    
    ### Results for trial linkage outcomes all combined
    # for train valid and test set linkage pd phase 1
    result1 =eval_TOP(linkage_pd, top_data_path, 'phase_I', test_only = False)
    # for train valid and test set linkage pd phase2
    result2 =eval_TOP(linkage_pd, top_data_path, 'phase_II', test_only = False)
    # for train valid and test set linkage pd phase3
    result3 =eval_TOP(linkage_pd, top_data_path, 'phase_III', test_only = False)
    # for train valid and test set linkage pd all
    result4 =eval_TOP(linkage_pd, top_data_path, 'all', test_only = False)

    # combine all the results and save it as a csv
    results = pd.DataFrame([result1,result2,result3,result4])
    # round all the values to .4f
    results = results.round(4)
    results['phase'] = ['phase_I','phase_II','phase_III','all']
    results.to_csv(os.path.join(save_path,'trial_linkage_TOP_train_valid_test_split_results.csv'),index=False)
    
    
# 5