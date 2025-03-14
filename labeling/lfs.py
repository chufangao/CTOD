import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, cohen_kappa_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import argparse
from snorkel.labeling.model import LabelModel, MajorityLabelVoter
from collections import Counter

def reorder_columns(df, cols_in_front):
    """Reorder columns in a pandas dataframe so that the columns in cols_in_front are in front.
    """
    columns = list(df.columns)
    for col in cols_in_front:
        columns.remove(col)
    columns = cols_in_front + columns
    return df[columns]

def lf_results_reported(path):
    df = pd.read_csv(path + 'calculated_values.txt', sep='|', low_memory=False)
    df.dropna(subset=['were_results_reported'], inplace=True)
    df['lf'] = df['were_results_reported'] == 't'
    df['lf'] = df['lf'].astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_num_sponsors(path, quantile=.5):
    df = pd.read_csv(path + 'sponsors.txt', sep='|',low_memory=False)
    df.dropna(subset=['name'], inplace=True)
    df = df.groupby('nct_id')['name'].count().reset_index()
    df['lf'] = df['name'] > df['name'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_num_patients(path, quantile=.5):
    df = pd.read_csv(path + 'outcome_counts.txt', sep='|', low_memory=False)    
    df.dropna(subset=['count'], inplace=True)
    df = df.groupby('nct_id').sum().reset_index() # pd df (NCTID, values, num_patients)
    df['lf'] = df['count'] > df['count'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_patient_drop(path, quantile=.5):
    # patient dropout
    df = pd.read_csv(os.path.join(path, 'drop_withdrawals.txt'), sep='|',low_memory=False)
    df.dropna(subset=['count'], inplace=True)
    df = df.groupby('nct_id').sum().reset_index() # pd df (NCTID, values, patient_drop)
    df['lf'] = df['count'] < df['count'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_sites(path, quantile=.5):
    # sites
    df = pd.read_csv(os.path.join(path, 'facilities.txt'), sep='|',low_memory=False)
    df.dropna(subset=['name'], inplace=True)
    df = df.groupby('nct_id')['name'].count().sort_values(ascending=False).reset_index()
    df = df.groupby('nct_id').mean().reset_index() # pd df (NCTID, values, sites)
    df['lf'] = df['name'] > df['name'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_pvalues(path): # any p-value sig is good
    # pvalues
    df = pd.read_csv(os.path.join(path, 'outcome_analyses.txt'), sep='|', low_memory=False)
    outcomes_df = pd.read_csv(os.path.join(path, 'outcomes.txt'), sep='|')
    primary_outcomes = outcomes_df[outcomes_df['outcome_type'].str.lower()=='primary']
    df = df[df['outcome_id'].isin(primary_outcomes['id'])]
    df.dropna(subset=['p_value'], inplace=True)

    df['lf'] = df['p_value'] < .05 
    df = df.groupby('nct_id')[['lf', 'p_value']].mean().reset_index() # multiple pvalues per nct_id
    df['lf'] = df['lf'] > 0 # any p-value sig is good
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_update_more_recent(path, quantile=.5):
    # if last update is more much recent than completion date, then it's bad
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

def lf_death_ae(path, quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|', low_memory=False)
    df = df[df['event_type'] == 'deaths'].fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_serious_ae(path, quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|', low_memory=False)
    df = df[df['event_type'] == 'serious'].fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_all_ae(path, quantile=.5):
    df = pd.read_csv(path+'reported_event_totals.txt', sep = '|', low_memory=False).fillna(0)
    df = df.groupby('nct_id')['subjects_affected'].sum().reset_index()
    df['lf'] = df['subjects_affected'] <= df['subjects_affected'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_status(path):
    df = pd.read_csv(path+'studies.txt', sep='|', low_memory=False)
    df['lf'] = -1
    # lower case all status and replace '_' with ' '
    df['overall_status'] = df['overall_status'].str.lower().str.replace('_', ' ')
    df.loc[df['overall_status'].isin(['terminated', 'withdrawn', 'suspended', 'withheld', 'no longer available', 'temporarily not available']),['lf']] = 0
    df.loc[df['overall_status'].isin(['approved for marketing']),['lf']] = 1
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def lf_amendments(path, quantile=.5):
    df = pd.read_csv(path, low_memory=False)
    df.dropna(subset=['amendment_counts'], inplace=True)
    df['lf'] = df['amendment_counts'] > df['amendment_counts'].quantile(quantile)
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
    
def lf_stock_price(path):
    df = pd.read_csv(path, low_memory=False)
    df.dropna(subset=['Slope'], inplace=True)
    df['lf'] = df['Slope'] > 0
    df['lf'] = df['lf'].fillna(-1).astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
    
def lf_linkage(path):
    df = pd.read_csv(path, low_memory=False)
    df.rename(columns={'nctid': 'nct_id'}, inplace=True)
    df['lf'] = 0
    df.loc[df['outcome']=='Failure',['lf']] = 0
    df.loc[df['outcome']=='Not sure',['lf']] = -1
    df.loc[df['outcome']=='Success', ['lf']] = 1
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

# def lf_news_headlines(path, path2, quantile=.5):
#     # df = pd.read_csv(path)
#     # df = df.fillna(0)qunatile
#     # sums = df['top_1_sim'] + df['top_2_sim'] + df['top_3_sim']
#     # df['lf'] = sums > sums.quantile(.5)
#     # df['lf'] = df['lf'].astype('int')
#     # df = reorder_columns(df, ['nct_id', 'lf'])
#     news_df = pd.read_csv(path2, low_memory=False)
#     df = pd.read_csv(path, low_memory=False)
#     for i in range(1, 4):
#         thresh = df[f'top_{i}_sim'].quantile(quantile)
#         # print(thresh)
#         df['top_'+str(i)] = df.apply(lambda x: news_df.iloc[int(x[f'top_{i}'])]['sentiment'] if x[f'top_{i}_sim'] > thresh else 'None', axis=1)

#     # get mode of top 3 sentiments if not None
#     df['valid_sentiments'] = df[['top_1', 'top_2', 'top_3']].apply(lambda x: [i for i in x if i != 'None' and i != 'Neutral'], axis=1)
#     # df['mode'] = df['valid_sentiments'].apply(lambda x: max(set(x), key=x.count) if x else 'None')
#     df['mode'] = df['valid_sentiments'].apply(lambda x: Counter(x).most_common(1)[0][0] if len(x)>0 else 'None')
#     df['lf'] = -1
#     df.loc[df['mode']=='Negative', ['lf']] = 0
#     df.loc[df['mode']=='Positive', ['lf']] = 1
#     df['lf'] = df['lf'].astype('int')
#     df = reorder_columns(df, ['nct_id', 'lf'])
#     return df
def lf_news_headlines(path):
    df = pd.read_csv(path, low_memory=False)
    df['lf'] = df['lf'].astype('int')
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df
           
def lf_gpt(path):
    df = pd.read_csv(path, low_memory=False)
    df['outcome'].unique()
    df['lf'] = -1
    df.loc[df['outcome']=='Success', ['lf']] = 1
    # df.loc[df['outcome']=='Not sure',['lf']] = 1
    # df.loc[df['outcome']=='Not sure',['lf']] = -1
    df.loc[df['outcome']=='Not sure',['lf']] = 0
    df.loc[df['outcome']=='Failure', ['lf']] = 0
    df.rename(columns={'nctid': 'nct_id'}, inplace=True)
    df = reorder_columns(df, ['nct_id', 'lf'])
    return df

def hint_train_lf(path):
    all_files = glob.glob(os.path.join(path, "phase*train.csv")) + glob.glob(os.path.join(path, "phase*valid.csv")) + glob.glob(os.path.join(path, "phase*test.csv"))
    # all_files = glob.glob(os.path.join(path, "phase*train.csv")) + glob.glob(os.path.join(path, "phase*valid.csv"))
    hint = pd.concat((pd.read_csv(f, low_memory=False) for f in all_files))
    hint.rename(columns={'nctid': 'nct_id'}, inplace=True)
    # print(hint['label'].value_counts())
    hint['lf'] = -1
    hint.loc[hint['label']==1, ['lf']] = 1
    hint.loc[hint['label']==0, ['lf']] = 0
    hint = reorder_columns(hint, ['nct_id', 'lf'])
    return hint

def get_lfs_by_name(func_name, quantile, 
                    path, 
                    LABELS_AND_TICKERS_PATH):
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
        return lf_amendments(quantile=quantile, path=LABELS_AND_TICKERS_PATH)
    else:
        print('func_name not found', func_name)
        return None

def get_lfs(lf_each_thresh_path,
            path, 
            HINT_PATH,
            GPT_PATH,
            LINKAGE_PATH,
            LABELS_AND_TICKERS_PATH,
            STUDIES_WITH_NEWS_PATH,
            no_hint=False):
    lf_thresh_df = pd.read_csv(lf_each_thresh_path, low_memory=False).sort_values(['lf', 'phase','acc'], ascending=False).astype(str)
    lf_thresh_df['best_thresh'] = 0
    for lf in lf_thresh_df['lf'].unique():
        for phase in ['1', '2', '3']:
            max = lf_thresh_df[(lf_thresh_df['phase']==phase) & (lf_thresh_df['lf']==lf)]['acc'].max()
            if eval(max) > .5:
                lf_thresh_df.loc[(lf_thresh_df['phase']==phase) & (lf_thresh_df['lf']==lf) & (lf_thresh_df['acc']==max), 'best_thresh'] = 1
    lf_thresh_df = lf_thresh_df[lf_thresh_df['best_thresh']==1]
    lf_thresh_df = lf_thresh_df.drop_duplicates(subset=['lf', 'phase']).reset_index(drop=True)
    
    status_lf = lf_status(path=path).iloc[:,:2] # only keep nct_id and lf
    hint_lf = hint_train_lf(path=HINT_PATH).iloc[:,:2] # only keep nct_id and lf
    gpt_lf = lf_gpt(path=GPT_PATH).iloc[:,:2] # only keep nct_id and lf
    linkage_lf = lf_linkage(path=LINKAGE_PATH).iloc[:,:2] # only keep nct_id and lf
    stock_price_lf = lf_stock_price(path=LABELS_AND_TICKERS_PATH).iloc[:,:2] # only keep nct_id and lf
    results_reported_lf = lf_results_reported(path=path).iloc[:,:2] # only keep nct_id and lf
    new_headlines_lf = lf_news_headlines(path=STUDIES_WITH_NEWS_PATH).iloc[:,:2]
    pvalues_lf = lf_pvalues(path=path).iloc[:,:2]

    known_lfs_list = [hint_lf,hint_lf,hint_lf, status_lf,status_lf, gpt_lf,gpt_lf, linkage_lf,linkage_lf, stock_price_lf, results_reported_lf, new_headlines_lf, pvalues_lf]
    df_names = ['hint_train', 'hint_train2', 'hint_train3', 'status','status2',  'gpt','gpt2', 'linkage','linkage2',  'stock_price', 'results_reported', 'new_headlines', 'pvalues']
    phase_dfs = []
    for phase in ['1', '2', '3']:
        phase_lfs = known_lfs_list.copy()
        phase_df_names = df_names.copy()
        for i, row in lf_thresh_df[lf_thresh_df['phase']==phase].iterrows():
            if no_hint == True:
                quantile = .5
            else:
                quantile = eval(row['quantile'])
            print("lf", row['lf'], "phase", row['phase'], "quantile", quantile)
            lf = get_lfs_by_name(row['lf'], quantile, path=path, LABELS_AND_TICKERS_PATH=LABELS_AND_TICKERS_PATH)
            phase_lfs.append(lf.iloc[:,:2]) # only keep nct_id and lf

            lf_name = str(row['lf'])
            occurence = len([x for x in phase_df_names if lf_name in x])
            if occurence > 0:
                lf_name = lf_name + str(occurence+1)
            phase_df_names.append(lf_name)

        all_ids = set() # set of all nct_ids
        for df in phase_lfs:
            all_ids = all_ids | set(df['nct_id'])
        all_df = pd.DataFrame(all_ids, columns=['nct_id']) # combine all dfs

        for i, df in enumerate(phase_lfs):
            # print(phase_df_names[i], all_df.columns, df.columns)
            all_df = pd.merge(all_df, df.rename(columns={'lf':phase_df_names[i]}), on='nct_id', how='left')
        all_df = all_df.fillna(-1)
        phase_dfs.append(all_df)

    return phase_dfs

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LF_EACH_THRESH_PATH', type=str, default='./lf_each_thresh.csv')
    parser.add_argument('--CTTI_PATH', type=str, default='../CTTI/')
    parser.add_argument('--HINT_PATH', type=str, default='./clinical-trial-outcome-prediction/data/')
    parser.add_argument('--GPT_PATH', type=str, default='../llm_prediction_on_pubmed/pubmed_gpt_outcomes.csv')
    parser.add_argument('--LINKAGE_PATH', type=str, default='../clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv')
    parser.add_argument('--LABELS_AND_TICKERS_PATH', type=str, default='../stock_price/labels_and_tickers.csv')
    parser.add_argument('--STUDIES_WITH_NEWS_PATH', type=str, default='../news_headlines/studies_with_news.csv')
    parser.add_argument('--CTO_GOLD_PATH', type=str, default='/srv/local/data/CTO/outcome_labels/final_cto_labels_2020_2024.csv"')
    parser.add_argument('--label_mode', type=str, default='DP')
    parser.add_argument('--get_thresholds', type=bool, default=False)
    args = parser.parse_args()
    print(args)

    cto_gold = pd.read_csv(args.CTO_GOLD_PATH)
    cto_gold.rename(columns={'labels': 'label'}, inplace=True)
    cto_gold = cto_gold[cto_gold['label'] != -1]
    print('cto_gold', cto_gold['label'].value_counts())

    # # all_files = glob.glob(os.path.join(path, "phase*train.csv")) + glob.glob(os.path.join(path, "phase*valid.csv")) + glob.glob(os.path.join(path, "phase*test.csv"))
    # all_files = glob.glob(os.path.join(args.HINT_PATH, "phase*test.csv"))
    # cto_gold = pd.concat((pd.read_csv(f, low_memory=False) for f in all_files))
    # cto_gold.rename(columns={'nctid': 'nct_id'}, inplace=True)
    # print('cto_gold', cto_gold['label'].value_counts())


    study_df = pd.read_csv(os.path.join(args.CTTI_PATH, 'studies.txt'), sep='|', low_memory=False)
    study_df.dropna(subset=['phase'], inplace=True)
    # only keep drug and biological interventions
    intervention_df = pd.read_csv(os.path.join(args.CTTI_PATH, 'interventions.txt'), sep='|', low_memory=False)
    intervention_df = intervention_df[intervention_df['intervention_type'].str.lower().isin(['drug', 'biological'])]
    study_df = study_df[study_df['nct_id'].isin(set(intervention_df['nct_id']))]
    # only keep completed studies
    study_df = study_df[study_df['overall_status'].str.lower().isin(['terminated', 'withdrawn', 'suspended', 'withheld', 'no longer available', 'temporarily not available', 'approved for marketing', 'completed'])]
    print(f"studies phase 1: {study_df[study_df['phase'].str.contains('1')].shape[0]}, phase 2: {study_df[study_df['phase'].str.contains('2')].shape[0]}, phase 3: {study_df[study_df['phase'].str.contains('3')].shape[0]}")

    hint = hint_train_lf(args.HINT_PATH)
    hint = hint[hint['nct_id'].isin(set(study_df['nct_id']))]
    hint = hint[~hint['nct_id'].isin(set(cto_gold['nct_id']))]
    print('hint', hint['lf'].value_counts())

    if args.get_thresholds == True:
        # quantile_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9,]
        quantile_list = np.linspace(.05, .95, 19)

        # funcs_all_name = ['num_sponsors', 'num_patients', 'patient_drop', 'sites', 'pvalues', 'update_more_recent', 'death_ae', 'serious_ae', 'all_ae', 'status', 'amendments', 'news_headlines']
        funcs_all_name = ['num_sponsors', 'num_patients', 'patient_drop', 'sites', 'update_more_recent', 'death_ae', 'serious_ae', 'all_ae', 'amendments']

        output = []
        for quantile in tqdm(quantile_list):
            funcs_all = [
                lf_num_sponsors(quantile=quantile, path=args.CTTI_PATH), 
                lf_num_patients(quantile=quantile, path=args.CTTI_PATH),
                lf_patient_drop(quantile=quantile, path=args.CTTI_PATH),
                lf_sites(quantile=quantile, path=args.CTTI_PATH), 
                lf_update_more_recent(quantile=quantile, path=args.CTTI_PATH), 
                lf_death_ae(quantile=quantile, path=args.CTTI_PATH), 
                lf_serious_ae(quantile=quantile, path=args.CTTI_PATH),
                lf_all_ae(quantile=quantile, path=args.CTTI_PATH),
                lf_amendments(quantile=quantile, path=args.LABELS_AND_TICKERS_PATH),
                ]

            for i in range(len(funcs_all)):
                for phase in ['1', '2', '3']:
                    names = funcs_all_name[i]

                    hint_phase_df = hint[hint['phase'].str.contains(phase)]
                    funcs = funcs_all[i][funcs_all[i]['nct_id'].isin(hint_phase_df['nct_id'])]
                    value_dict = funcs['lf'].value_counts().to_dict()

                    for key in [-1.0, 0.0, 1.0]:
                        if key not in value_dict:
                            value_dict[key] = 0
                    if value_dict[1.0] + value_dict[0.0] == 0:
                        positive_perc = 0
                        print(f"no positive or negative labels for {names} in phase {phase}")
                        continue
                        # quit()
                    else:
                        positive_perc = value_dict[1.0] / (value_dict[1.0] + value_dict[0.0])

                    len_all_trials = study_df[study_df['phase'].str.contains(phase)].shape[0]
                    coverage = sum([value_dict[k] for k in value_dict.keys() if k!=-1.0]) / len_all_trials

                    # print(hint_phase_df.copy().columns, funcs.columns)
                    combined = pd.merge(hint_phase_df.copy().drop(columns=['lf']), funcs, on='nct_id', how='left')
                    combined = combined[combined['lf'] != -1].dropna(subset=['lf'])

                    output.append(f"{names}, {phase}, {quantile}, {value_dict[-1.0]}, {value_dict[0.0]}, {value_dict[1.0]}, {positive_perc}, {coverage}, {accuracy_score(combined['label'], combined['lf'])}, {cohen_kappa_score(combined['label'], combined['lf'])}")
                    # print(output[-1])
        df = pd.DataFrame([x.split(',') for x in output], columns=['lf', 'phase', 'quantile', '-1.0', '0.0', '1.0', 'positive_perc', 'coverage', 'acc', 'ck'])
        df.to_csv(args.LF_EACH_THRESH_PATH, index=False)

    # ==== load best thresholds ====
    no_hint = True if args.label_mode != 'DP' else False
    df_list = get_lfs(lf_each_thresh_path=args.LF_EACH_THRESH_PATH,
                                 path=args.CTTI_PATH, 
                                 HINT_PATH=args.HINT_PATH,
                                 GPT_PATH=args.GPT_PATH,
                                 LINKAGE_PATH=args.LINKAGE_PATH,
                                 LABELS_AND_TICKERS_PATH=args.LABELS_AND_TICKERS_PATH,
                                 STUDIES_WITH_NEWS_PATH=args.STUDIES_WITH_NEWS_PATH,
                                 no_hint=no_hint)
    

    # ==== fit dp ====
    # bad_top_test_df = pd.read_csv('./manual_labels/mismatched_status.csv').rename(columns={'nctid': 'nct_id'})
    all_combineds = []
    all_combined_full = []
    all_phases = ['1', '2', '3']

    print("phase, acc, f1, prauc, rocauc, kappa")
    for i in [0,1,2]:
        phase = all_phases[i]
        df2 = df_list[i].copy()
        # subset on phase

        df2 = df2[df2['nct_id'].isin(study_df[study_df['phase'].str.contains(phase)]['nct_id'])]
        df2['pred'] = -1
        # df2 = df2.sort_values('nct_id')
        # subset on intervention type
        L = df2.iloc[:,1:].values.astype('int')
        if args.label_mode in ['RF', 'SVM', 'LR']:
            X1, Y = L[L[:,0]!=-1,3:], L[L[:,0]!=-1,0] # 3: is the first lf that's not hint, subset on where hint is valid
            X2 = L[:,3:] # predict on all
            if args.label_mode == 'RF':
                model = RandomForestClassifier(n_estimators=1000, max_depth=4, random_state=0)
            elif args.label_mode == 'SVM':
                model = CalibratedClassifierCV()
            elif args.label_mode == 'LR':
                model = LogisticRegression(random_state=0, max_iter=2000)
            # pred = XGBClassifier(n_estimators=100, max_depth=5, random_state=0).fit(X1,Y).predict(X2)
            # model = LinearSVC(random_state=0)
            # model = LogisticRegression(random_state=0, max_iter=2000)
            label_model_pred_proba = model.fit(X1,Y).predict_proba(X2)[:,1]
            label_model_pred = model.predict(X2)
        elif args.label_mode == 'DP':
            positive_props = [.4, .4, .4]
            lrs = [.01, .01, .01]
            label_model = LabelModel(verbose=False, cardinality=2)
            # label_model = MajorityLabelVoter(cardinality=2)
            positive_prop = positive_props[i]
            label_model.fit(L, class_balance=[1-positive_prop, positive_prop], seed=0, lr=lrs[i], n_epochs=300)
            label_model_pred_proba = label_model.predict_proba(L)[:,1]
            label_model_pred = label_model.predict(L)
        elif args.label_mode == 'DP_nohint':
            positive_props = [.4, .4, .4]
            lrs = [.01, .01, .01]
            label_model = LabelModel(verbose=False, cardinality=2)
            # label_model = MajorityLabelVoter(cardinality=2)
            positive_prop = positive_props[i]
            label_model.fit(L[:,3:], class_balance=[1-positive_prop, positive_prop], seed=0, lr=lrs[i], n_epochs=300)
            label_model_pred_proba = label_model.predict_proba(L[:,3:])[:,1]
            label_model_pred = label_model.predict(L[:,3:])

        # apply status lf
        status_lf = lf_status(path=args.CTTI_PATH)
        status_lf = status_lf[status_lf['lf']!=-1]
        status_mapping = status_lf[['nct_id', 'lf']].set_index('nct_id').to_dict()['lf']
        df2['pred'] = df2.apply(lambda x: status_mapping[x['nct_id']] if x['nct_id'] in status_mapping else x['pred'], axis=1)
        print(df2['pred'].value_counts())

        # apply pvalue mapping where pvalue < 0.05 (conservative, some studies may not have pvalues but are still positive)
        pvalue_lf = lf_pvalues(path=args.CTTI_PATH)
        pvalue_lf = pvalue_lf[pvalue_lf['lf']==1]
        pvalue_mapping = pvalue_lf[['nct_id', 'lf']].set_index('nct_id').to_dict()['lf']
        # map lf == -1 to pvalue lf
        df2.loc[df2['pred']==-1, 'pred'] = df2.loc[df2['pred']==-1].apply(lambda x: pvalue_mapping[x['nct_id']] if x['nct_id'] in pvalue_mapping else x['pred'], axis=1)
        print(df2['pred'].value_counts())

        df2['pred_proba'] = df2['pred']
        mask = df2['pred'] == -1

        # apply labelmodel pred where pred == -1
        df2.loc[mask, 'pred_proba'] = label_model_pred_proba[mask]
        df2.loc[mask, 'pred'] = label_model_pred[mask]
        print(df2['pred'].value_counts())

        # append specific phase dfs
        all_combined_full.append(df2)

        gold_df = cto_gold[cto_gold['phase'].str.contains(phase)]
        # gold_df['label'] = gold_df['label'].apply(lambda x: status_mapping[x] if x in status_mapping else x)
        # gold_df = gold_df[~gold_df['nct_id'].isin(bad_top_test_df['nct_id'])]
        combined = pd.merge(gold_df, df2, on='nct_id', how='left')
        combined = combined.dropna(subset=['pred'])
        combined = combined[combined['pred'] != -1]
        # print(phase, hint_subset.shape, combined.shape)
        print(phase,',', accuracy_score(combined['label'], combined['pred']), ',',
            f1_score(combined['label'], combined['pred']), ',', 
            average_precision_score(combined['label'], combined['pred']), ',',
            roc_auc_score(combined['label'], combined['pred']), ',',
            cohen_kappa_score(combined['label'], combined['pred']))

        all_combineds.append(combined)

    combined = pd.concat(all_combineds)
    print('all',',', accuracy_score(combined['label'], combined['pred']), ',',
        f1_score(combined['label'], combined['pred']), ',', 
        average_precision_score(combined['label'], combined['pred']), ',',
        roc_auc_score(combined['label'], combined['pred']), ',',
        cohen_kappa_score(combined['label'], combined['pred']))
    
    # save results
    all_combined_full[0].to_csv(f'phase1_{args.label_mode.lower()}.csv', index=False)
    all_combined_full[1].to_csv(f'phase2_{args.label_mode.lower()}.csv', index=False)
    all_combined_full[2].to_csv(f'phase3_{args.label_mode.lower()}.csv', index=False)

