import pandas as pd
import os
import glob
import numpy as np
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from sklearn.metrics import classification_report
from collections import Counter

def reorder_columns(df, cols_in_front):
    """Reorder columns in a pandas dataframe so that the columns in cols_in_front are in front.
    """
    columns = list(df.columns)
    for col in cols_in_front:
        columns.remove(col)
    columns = cols_in_front + columns
    return df[columns]

def lf_results_reported(path='../CTTI/'):
    df = pd.read_csv(path + 'calculated_values.txt', sep='|', low_memory=False)
    df['lf'] = df['were_results_reported'] == 't'
    df['lf'] = df['lf'].astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_num_sponsors(path='../CTTI/', quantile=.5):
    df = pd.read_csv(path + 'sponsors.txt', sep='|')
    df = df.groupby('nct_id')['name'].count().reset_index()
    df['lf'] = df['name'] > df['name'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_num_patients(path='../CTTI/', quantile=.5):
    df = pd.read_csv(path + 'outcome_counts.txt', sep='|', low_memory=False)    
    df = df.groupby('nct_id').sum().reset_index() # pd df (NCTID, values, num_patients)
    df['lf'] = df['count'] > df['count'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_patient_drop(path='../CTTI/', quantile=.5):
    # patient dropout
    df = pd.read_csv(os.path.join(path, 'drop_withdrawals.txt'), sep='|')
    df = df.groupby('nct_id').sum().reset_index() # pd df (NCTID, values, patient_drop)
    df['lf'] = df['count'] < df['count'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_sites(path='../CTTI/', quantile=.5):
    # sites
    df = pd.read_csv(os.path.join(path, 'facilities.txt'), sep='|')
    df = df.groupby('nct_id')['name'].count().sort_values(ascending=False).reset_index()
    df = df.groupby('nct_id').mean().reset_index() # pd df (NCTID, values, sites)
    df['lf'] = df['name'] > df['name'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_pvalues(path='../CTTI/', quantile=.5):
    # pvalues
    path = '../CTTI/'
    df = pd.read_csv(os.path.join(path, 'outcome_analyses.txt'), sep='|', low_memory=False)
    df['lf'] = df['p_value'] < .05 # 89406
    df = df.groupby('nct_id')[['lf', 'p_value']].mean().reset_index() # multiple pvalues per nct_id
    df['lf'] = df['lf'] > df['lf'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_update_more_recent(path='../CTTI/', quantile=.5): #TODO clarify what this does
    df = pd.read_csv(os.path.join(path, 'studies.txt'), sep='|', low_memory=False)
    df['last_update_submitted_date'] = pd.to_datetime(df['last_update_submitted_date'])
    df['completion_date'] = pd.to_datetime(df['completion_date'])
    df['update_days'] = (df['last_update_submitted_date'] - df['completion_date']).dt.days
    median = df['update_days'].quantile(quantile) 
    # print(median)
    df['lf'] = df['update_days'].apply(lambda x: x < median if pd.notna(x) else x)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf']) 
    return df

def lf_death_ae(path='../CTTI/', quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|')
    df = df[df['event_type'] == 'deaths'].fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_serious_ae(path='../CTTI/', quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|')
    df = df[df['event_type'] == 'serious'].fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_all_ae(path='../CTTI/', quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|').fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_status(path='../CTTI/'):
    df = pd.read_csv(path+'studies.txt', sep='|')
    df['lf'] = -1
    df.loc[df['overall_status'].isin(['Terminated', 'Withdrawn', 'Suspended', 'Withheld', 'No longer available', 'Temporarily not available']),['lf']] = 0
    df.loc[df['overall_status'].isin(['Approved for marketing']),['lf']] = 1
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_amendments(path='../stock_price/labels_and_tickers.csv', quantile=.5):
    df = pd.read_csv(path)
    df['lf'] = df['amendment_counts'] > df['amendment_counts'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
    
def lf_stock_price(path='../stock_price/labels_and_tickers.csv'):
    df = pd.read_csv(path)
    df['lf'] = df['Slope'] > 0
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
    
def lf_linkage(path='../clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv'):
    df = pd.read_csv(path)
    df.rename(columns={'nctid': 'nct_id'}, inplace=True)
    df['lf'] = 0
    # df.loc[df['outcome']=='Not sure',['lf']] = 1
    # df.loc[df['outcome']=='Not sure',['lf']] = -1
    df.loc[df['outcome']=='Not sure',['lf']] = 0
    df.loc[df['outcome']=='Success', ['lf']] = 1
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_news_headlines(path='../news_headlines/studies_with_news.csv',path2='../news_headlines/news.csv', quantile=.5):
    # df = pd.read_csv(path)
    # df = df.fillna(0)
    # sums = df['top_1_sim'] + df['top_2_sim'] + df['top_3_sim']
    # df['lf'] = sums > sums.quantile(.5)
    # df['lf'] = df['lf'].astype('int')
    # df = reorder_columns(df, ['nct_id', 'lf'])
    news_df = pd.read_csv(path2)
    df = pd.read_csv(path)
    for i in range(1, 4):
        thresh = df[f'top_{i}_sim'].quantile(quantile)
        # print(thresh)
        df['top_'+str(i)] = df.apply(lambda x: news_df.iloc[int(x[f'top_{i}'])]['sentiment'] if x[f'top_{i}_sim'] > thresh else 'None', axis=1)

    # get mode of top 3 sentiments if not None
    df['valid_sentiments'] = df[['top_1', 'top_2', 'top_3']].apply(lambda x: [i for i in x if i != 'None' and i != 'Neutral'], axis=1)
    # df['mode'] = df['valid_sentiments'].apply(lambda x: max(set(x), key=x.count) if x else 'None')
    df['mode'] = df['valid_sentiments'].apply(lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else 'None')
    df['lf'] = -1
    df.loc[df['mode']=='Negative', ['lf']] = 0
    df.loc[df['mode']=='Positive', ['lf']] = 1
    df['lf'] = df['lf'].astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
           
def lf_gpt(path='../llm_prediction_on_pubmed/pubmed_gpt_outcomes.csv'):
    df = pd.read_csv(path)
    df['outcome'].unique()
    df['lf'] = -1
    df.loc[df['outcome']=='Success', ['lf']] = 1
    # df.loc[df['outcome']=='Not sure',['lf']] = 1
    # df.loc[df['outcome']=='Not sure',['lf']] = -1
    df.loc[df['outcome']=='Not sure',['lf']] = -1
    df.loc[df['outcome']=='Failure', ['lf']] = 0
    df.rename(columns={'nctid': 'nct_id'}, inplace=True)           
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def hint_train_lf(path='./clinical-trial-outcome-prediction/data/'):
    all_files = glob.glob(os.path.join(path, "phase*train.csv")) + glob.glob(os.path.join(path, "phase*valid.csv"))
    # all_files = glob.glob(os.path.join(path, "phase*test.csv"))
    hint = pd.concat((pd.read_csv(f) for f in all_files))
    hint.rename(columns={'nctid': 'nct_id'}, inplace=True)
    # print(hint['label'].value_counts())
    hint['lf'] = -1
    hint.loc[hint['label']==1, ['lf']] = 1
    hint.loc[hint['label']==0, ['lf']] = 0
    hint = reorder_columns(hint, ['nct_id', 'lf'])
    return hint

def get_lfs_by_name(func_name, quantile, path='../CTTI/'):
    # funcs_all_name = ['num_sponsors', 'num_patients', 'patient_drop', 'sites', 'pvalues', 'update_more_recent', 'death_ae', 'serious_ae', 'all_ae', 'amendments', 'news_headlines']
    # funcs_all = [lf_num_sponsors(quantile=quantile), 
    #              lf_num_patients(quantile=quantile), 
    #              lf_patient_drop(quantile=quantile), 
    #              lf_sites(quantile=quantile), 
    #              lf_pvalues(quantile=quantile), 
    #              lf_update_more_recent(quantile=quantile), 
    #              lf_death_ae(quantile=quantile), 
    #              lf_serious_ae(quantile=quantile), 
    #              lf_all_ae(quantile=quantile), 
    #              lf_amendments(quantile=quantile), 
    #              lf_news_headlines(quantile=quantile)]
    if func_name == 'num_sponsors':
        return lf_num_sponsors(path=path, quantile=quantile)
    elif func_name == 'num_patients':
        return lf_num_patients(path=path, quantile=quantile)
    elif func_name == 'patient_drop':
        return lf_patient_drop(path=path, quantile=quantile)
    elif func_name == 'sites':
        return lf_sites(path=path, quantile=quantile)
    elif func_name == 'pvalues':
        return lf_pvalues(path=path, quantile=quantile)
    elif func_name == 'update_more_recent':
        return lf_update_more_recent(path=path, quantile=quantile)
    elif func_name == 'death_ae':
        return lf_death_ae(path=path, quantile=quantile)
    elif func_name == 'serious_ae':
        return lf_serious_ae(path=path, quantile=quantile)
    elif func_name == 'all_ae':
        return lf_all_ae(path=path, quantile=quantile)
    elif func_name == 'amendments':
        return lf_amendments(quantile=quantile)
    elif func_name == 'news_headlines':
        return lf_news_headlines(quantile=quantile)
    else:
        return None

def get_lfs(path='../CTTI/', lf_each_thresh_path='./lf_each_thresh.csv'):
    lf_thresh_df = pd.read_csv(lf_each_thresh_path).sort_values(['lf', 'phase','acc'], ascending=False).astype(str)
    lf_thresh_df['best_thresh'] = 0
    for lf in lf_thresh_df['lf'].unique():
        for phase in ['1', '2', '3']:
            max = lf_thresh_df[(lf_thresh_df['phase']==phase) & (lf_thresh_df['lf']==lf)]['acc'].max()
            if eval(max) > .5:
                lf_thresh_df.loc[(lf_thresh_df['phase']==phase) & (lf_thresh_df['lf']==lf) & (lf_thresh_df['acc']==max), 'best_thresh'] = 1
    lf_thresh_df = lf_thresh_df[lf_thresh_df['best_thresh']==1]
    lf_thresh_df = lf_thresh_df.drop_duplicates(subset=['lf', 'phase']).reset_index(drop=True)
    
    status_lf = lf_status(path=path)
    hint_lf = hint_train_lf()
    gpt_lf = lf_gpt()
    linkage_lf = lf_linkage()
    stock_price_lf = lf_stock_price()
    results_reported_lf = lf_results_reported(path=path)
#     dfs = [\
#         results_reported_lf,
# #         lf_num_sponsors(path=path, 0.9),
#         lf_num_patients(path=path, quantile=.4), 
#         lf_patient_drop(path=path, quantile=.9), 
#         lf_sites(path=path,quantile=.9), 
#         lf_pvalues(path=path, quantile=.5),
#         lf_update_more_recent(path=path, quantile=.9),
#         lf_death_ae(path=path, quantile=.5),
# #         lf_serious_ae(path=path,),
# #         lf_all_ae(path=path),
#         status_lf,
#         lf_amendments(quantile=.1),
#         stock_price_lf,
#         linkage_lf,
#         linkage_lf,
#         lf_news_headlines(quantile=.5),
#         gpt_lf,
#         gpt_lf,
#         hint_lf,
#         hint_lf,
#         hint_lf
#         ]
    known_lfs_list = [hint_lf,hint_lf,hint_lf, status_lf,status_lf, gpt_lf,gpt_lf, linkage_lf,linkage_lf, stock_price_lf, results_reported_lf]
    phase_dfs = []
    for phase in ['1', '2', '3']:
        phase_lfs = known_lfs_list.copy()
        # print(lf_thresh_df[lf_thresh_df['phase']==phase]['lf'])
        # continue
        for i, row in lf_thresh_df[lf_thresh_df['phase']==phase].iterrows():
            lf = get_lfs_by_name(row['lf'], eval(row['qunatile']), path=path)
            phase_lfs.append(lf)

        all_ids = set() # set of all nct_ids
        for df in phase_lfs:
            all_ids = all_ids | set(df['nct_id'])
        all_df = pd.DataFrame(all_ids, columns=['nct_id']) # combine all dfs
        for i, df in enumerate(phase_lfs):
            all_df = pd.merge(all_df, df.iloc[:,:2].rename(columns={'lf':'lf'+str(i)}), on='nct_id', how='left')
        all_df = all_df.fillna(-1)
        phase_dfs.append(all_df)

    return phase_dfs, status_lf