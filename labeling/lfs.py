import pandas as pd
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, cohen_kappa_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
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
    # df = df.fillna(0)qunatile
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

def get_lfs_by_name(func_name, quantile, 
                    path='../CTTI/', 
                    LABELS_AND_TICKERS_PATH='../stock_price/labels_and_tickers.csv',
                    STUDIES_WITH_NEWS_PATH='../news_headlines/studies_with_news.csv',
                    NEWS_PATH='../news_headlines/news.csv'):
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
    elif func_name == 'news_headlines':
        return lf_news_headlines(quantile=quantile, path=STUDIES_WITH_NEWS_PATH, path2=NEWS_PATH)
    else:
        print('func_name not found', func_name)
        return None

def get_lfs(lf_each_thresh_path='./lf_each_thresh.csv',
            path='../CTTI/', 
            HINT_PATH='./clinical-trial-outcome-prediction/data/',
            GPT_PATH='../llm_prediction_on_pubmed/pubmed_gpt_outcomes.csv',
            LINKAGE_PATH='../clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv',
            LABELS_AND_TICKERS_PATH='../stock_price/labels_and_tickers.csv',
            STUDIES_WITH_NEWS_PATH='../news_headlines/studies_with_news.csv',
            NEWS_PATH='../news_headlines/news.csv'):
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
    hint_lf = hint_train_lf(path=HINT_PATH)
    gpt_lf = lf_gpt(path=GPT_PATH)
    linkage_lf = lf_linkage(path=LINKAGE_PATH)
    stock_price_lf = lf_stock_price(path=LABELS_AND_TICKERS_PATH)
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
            print(row['lf'], row['phase'], row['quantile'])
            lf = get_lfs_by_name(row['lf'], eval(row['quantile']), path=path, 
                                 LABELS_AND_TICKERS_PATH=LABELS_AND_TICKERS_PATH,
                                 STUDIES_WITH_NEWS_PATH=STUDIES_WITH_NEWS_PATH,
                                 NEWS_PATH=NEWS_PATH)
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--LF_EACH_THRESH_PATH', type=str, default='./lf_each_thresh.csv')
    parser.add_argument('--CTTI_PATH', type=str, default='../CTTI/')
    parser.add_argument('--HINT_PATH', type=str, default='./clinical-trial-outcome-prediction/data/')
    parser.add_argument('--GPT_PATH', type=str, default='../llm_prediction_on_pubmed/pubmed_gpt_outcomes.csv')
    parser.add_argument('--LINKAGE_PATH', type=str, default='../clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv')
    parser.add_argument('--LABELS_AND_TICKERS_PATH', type=str, default='../stock_price/labels_and_tickers.csv')
    parser.add_argument('--STUDIES_WITH_NEWS_PATH', type=str, default='../news_headlines/studies_with_news.csv')
    parser.add_argument('--NEWS_PATH', type=str, default='../news_headlines/news.csv')
    args = parser.parse_args()
    print(args)

    all_files = glob.glob(os.path.join(args.HINT_PATH, "phase*train.csv")) + glob.glob(os.path.join(args.HINT_PATH, "phase*valid.csv"))
    hint = pd.concat((pd.read_csv(f) for f in all_files))
    hint.rename(columns={'nctid': 'nct_id'}, inplace=True)
    print(hint['label'].value_counts())

    # study_df = pd.read_csv('../CTTI/studies.txt', sep='|')
    # study_df.dropna(subset=['phase'], inplace=True)
    # study_df[study_df['phase'].str.contains('1')].shape[0], study_df[study_df['phase'].str.contains('2')].shape[0], study_df[study_df['phase'].str.contains('3')].shape[0]
    phase_1_sum, phase_2_sum, phase_3_sum = 60549, 76972, 44087

    # print('lf, phase, qunatile, -1.0, 0.0, 1.0, prop, coverage, acc, ck')
    quantile_list = [.1, .2, .3, .4, .5, .6, .7, .8, .9,]

    output = []
    for quantile in tqdm(quantile_list):
        funcs_all = [lf_num_sponsors(quantile=quantile, path=args.CTTI_PATH), 
                    lf_num_patients(quantile=quantile, path=args.CTTI_PATH), 
                    lf_patient_drop(quantile=quantile, path=args.CTTI_PATH), 
                    lf_sites(quantile=quantile, path=args.CTTI_PATH), 
                    lf_pvalues(quantile=quantile, path=args.CTTI_PATH), 
                    lf_update_more_recent(quantile=quantile, path=args.CTTI_PATH), 
                    lf_death_ae(quantile=quantile, path=args.CTTI_PATH), 
                    lf_serious_ae(quantile=quantile, path=args.CTTI_PATH), 
                    lf_all_ae(quantile=quantile, path=args.CTTI_PATH), 
                    # lf_status(path=args.CTTI_PATH), 
                    lf_amendments(quantile=quantile, path=args.LABELS_AND_TICKERS_PATH), 
                    lf_news_headlines(quantile=quantile, path=args.STUDIES_WITH_NEWS_PATH, path2=args.NEWS_PATH)]
        # funcs_all_name = ['num_sponsors', 'num_patients', 'patient_drop', 'sites', 'pvalues', 'update_more_recent', 'death_ae', 'serious_ae', 'all_ae', 'status', 'amendments', 'news_headlines']
        funcs_all_name = ['num_sponsors', 'num_patients', 'patient_drop', 'sites', 'pvalues', 'update_more_recent', 'death_ae', 'serious_ae', 'all_ae', 'amendments', 'news_headlines']

        for i in range(len(funcs_all)):
            for phase in ['1', '2', '3']:
                names = funcs_all_name[i] 

                labels_df = hint[hint['phase'].str.contains(phase)]
                funcs = funcs_all[i][funcs_all[i]['nct_id'].isin(labels_df['nct_id'])]
                value_counts = funcs['lf'].value_counts()
                value_dict = value_counts.to_dict()

                for key in [-1.0, 0.0, 1.0]:
                    if key not in value_dict:
                        value_dict[key] = 0

                positive_perc = value_dict[1.0] / (value_dict[1.0] + value_dict[0.0])
                if phase == '1':
                    len_all_trials = phase_1_sum
                elif phase == '2':
                    len_all_trials = phase_2_sum
                else:
                    len_all_trials = phase_3_sum
                coverage = sum([value_dict[k] for k in value_dict.keys() if k!=-1.0]) / len_all_trials

                combined = pd.merge(labels_df.copy(), funcs, on='nct_id', how='left')
                combined = combined[combined['lf'] != -1].dropna(subset=['lf'])

                output.append(f"{names}, {phase}, {quantile}, {value_dict[-1.0]}, {value_dict[0.0]}, {value_dict[1.0]}, {positive_perc}, {coverage}, \
                            {accuracy_score(combined['label'], combined['lf'])}, {cohen_kappa_score(combined['label'], combined['lf'])}")
                # print(output[-1])
    df = pd.DataFrame([x.split(',') for x in output], columns=['lf', 'phase', 'quantile', '-1.0', '0.0', '1.0', 'positive_perc', 'coverage', 'acc', 'ck'])
    df.to_csv(args.LF_EACH_THRESH_PATH, index=False)

    # ==== load best thresholds ====
    df_list, status_lf = get_lfs(lf_each_thresh_path=args.LF_EACH_THRESH_PATH,
                                 path=args.CTTI_PATH, 
                                 HINT_PATH=args.HINT_PATH,
                                 GPT_PATH=args.GPT_PATH,
                                 LINKAGE_PATH=args.LINKAGE_PATH,
                                 LABELS_AND_TICKERS_PATH=args.LABELS_AND_TICKERS_PATH,
                                 STUDIES_WITH_NEWS_PATH=args.STUDIES_WITH_NEWS_PATH,
                                 NEWS_PATH=args.NEWS_PATH)
    
    # all_files = glob.glob(os.path.join(path, "phase*train.csv")) + glob.glob(os.path.join(path, "phase*valid.csv")) + glob.glob(os.path.join(path, "phase*test.csv"))
    all_files = glob.glob(os.path.join(args.HINT_PATH, "phase*test.csv"))
    hint = pd.concat((pd.read_csv(f) for f in all_files))
    hint.rename(columns={'nctid': 'nct_id'}, inplace=True)
    print(hint['label'].value_counts())    

    # ==== fit dp ====
    # bad_top_test_df = pd.read_csv('./mismatched_status.csv').rename(columns={'nctid': 'nct_id'})
    positive_props = [.4, .5, .5]
    lrs = [.01, .01, .01]
    all_combineds = []
    all_combined_full = []
    all_phases = ['1', '2', '3']

    print("phase, acc, f1, prauc, rocauc, kappa")
    for i in [0,1,2]:
        phase = all_phases[i]
        df2 = df_list[i].copy()
        L = df2.iloc[:,1:].values.astype('int')

        label_model = LabelModel(verbose=False, cardinality=2)
        # label_model = MajorityLabelVoter(cardinality=2)

        positive_prop = positive_props[i]
        label_model.fit(L, class_balance=[1-positive_prop, positive_prop], seed=0, lr=lrs[i], n_epochs=200)
        pred = label_model.predict(L)
        df2['pred'] = pred.astype('int')

        df2 = df2.sort_values('nct_id')
        status_subset = status_lf[status_lf['lf']!=-1]
        status_subset_dict = dict(zip(status_subset['nct_id'], status_subset['lf']))
        df2['pred'] = df2.apply(lambda x: status_subset_dict[x['nct_id']] if x['nct_id'] in status_subset_dict else x['pred'], axis=1)

        all_combined_full.append(df2.copy())

        hint_subset = hint[hint['phase'].str.contains(phase)]
        # hint_subset = hint_subset[~hint_subset['nct_id'].isin(bad_top_test_df['nct_id'])]
        combined = pd.merge(hint_subset, df2, on='nct_id', how='left')
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
    all_combineds[0].to_csv('phase1_dp.csv', index=False)
    all_combineds[1].to_csv('phase2_dp.csv', index=False)
    all_combineds[2].to_csv('phase3_dp.csv', index=False)

