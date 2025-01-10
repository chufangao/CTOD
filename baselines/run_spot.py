# get current file path
import os
import sys
# get current file path
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'./PyTrial'))
from pytrial.tasks.trial_outcome import SPOT
from pytrial.tasks.trial_outcome.data import TrialOutcomeDataset
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
import json
import os
import pickle
import random
import shutil
from collections import defaultdict
import pandas as pd
import numpy as np
import torch
from torchmetrics import AUROC, AveragePrecision, F1Score, MetricCollection, StatScores
from tqdm import trange
from lightning import seed_everything

def bootstrap_testing(preds, target, metrics, bootstrap_num=20):
    results = []
    for _ in range(bootstrap_num):
        cur_result = {}
        idx = torch.randint_like(target, 0, len(target))
        cur_result["preds"] = preds[idx]
        cur_result["target"] = target[idx]

        metrics.update(**cur_result)
        results.append(metrics.compute())
        metrics.reset()

    result = {}
    if len(results) == 1:
        result = results[0]
    elif len(results) > 1:
        for key in results[0]:
            data = torch.stack([r[key] for r in results], dim=0).to(torch.float32)
            if "statscores" in key:
                result[f"{key}_min"] = data.min()
                result[f"{key}_max"] = data.max()
            result[f"{key}_mean"] = data.mean()
            result[f"{key}_std"] = data.std()
        for key in result:
            result[key] = result[key].item()
    return result

def get_random_seed():
    return random.randint(np.iinfo(np.uint32).min, np.iinfo(np.uint32).max)

#read your data
df1 = pd.read_csv('/home/trishad2/clinical-trial-outcome-prediction/data/HINT_train.csv')
df2 = pd.read_csv('/home/trishad2/clinical-trial-outcome-prediction/data/HINT_valid.csv')
df3 = pd.read_csv('/home/trishad2/clinical-trial-outcome-prediction/data/TOP_test.csv')

current_directory ='/home/trishad2/benchmark/'
dates = pd.read_csv(current_directory + 'submitted_dates.csv')


dates.rename(columns={'nct_id':'nctid'}, inplace=True)

df1 = pd.merge(df1, dates, on='nctid', how='inner')
df2 = pd.merge(df2, dates, on='nctid', how='inner')
df3 = pd.merge(df3, dates, on='nctid', how='inner')

titles = pd.read_csv(current_directory + 'titles.csv')

titles.rename(columns={'nct_id':'nctid'}, inplace=True)
df1 = pd.merge(df1, titles, on='nctid', how='inner')
df2 = pd.merge(df2, titles, on='nctid', how='inner')
df3 = pd.merge(df3, titles, on='nctid', how='inner')

#replace NaNs with empty strings in 'criteria' column
df1['criteria'] = df1['criteria'].fillna('')
df2['criteria'] = df2['criteria'].fillna('')
df3['criteria'] = df3['criteria'].fillna('')

print(df1.shape, df2.shape, df3.shape)

train_df = TrialOutcomeDataset(df1)
valid_df = TrialOutcomeDataset(df2)
test_df = TrialOutcomeDataset(df3)

target_trials = df3["nctid"].tolist()
print('length of target_trials: ',len(target_trials))

metrics = MetricCollection(
        {
            "F1": F1Score("binary"),
            "ROC-AUC": AUROC("binary"),
            "PR-AUC": AveragePrecision("binary"),
            "STAT": StatScores("binary"),
        }
    )

#fit_modal(datasets=datasets, metrics=metrics, model_name="spot", bootstrap_test=False)
seed = get_random_seed()
seed_everything(seed)
output_path="/home/trishad2/lifted/results/details/"
model_args = {"seed": seed, "output_dir": output_path, "learning_rate": 1e-3, "weight_decay": 0}


model = SPOT(**model_args)
model.fit(train_df, valid_df)


print('Predicting on test set')
preds = model.predict(test_df, target_trials=target_trials)
nctids =preds["nctid"]
target = torch.tensor(preds["label"])
predictions = torch.tensor(preds["pred"])

print(predictions, len(predictions))
print(target)

#create a dataframe with nctids, targets and predictions
df4 = pd.DataFrame({'nctid': nctids, 'label': target, 'prediction': predictions.flatten()})
print(df4.head())
print(df4.shape)
#add column phase from df3 to df4
df4 = pd.merge(df4, df3[['nctid', 'phase']], on='nctid', how='inner')

#save the predictions as csv
#df4.to_csv('/home/trishad2/lifted/tools/models/spot_results/HINT_test_preds_by_SPOT.csv', index=False)

#calculate metrics on the df4
print('F1: ' ,f1_score(df4['label'], (df4['prediction']>=0.5).astype(int)))
print('AP: ', average_precision_score(df4['label'], df4['prediction']))
print('ROC-AUC: ', roc_auc_score(df4['label'], df4['prediction']))

for phase in ['phase 1', 'phase 2', 'phase 3']:
    test_df_subset = df4[df4['phase'].str.lower().str.contains(phase)]

    print(phase)

    #F1
    print('F1: ' ,f1_score(test_df_subset['label'], (test_df_subset['prediction']>=0.5).astype(int)))

    #AP
    print('AP: ', average_precision_score(test_df_subset['label'], test_df_subset['prediction']))

    #ROC-AUC
    print('ROC-AUC: ', roc_auc_score(test_df_subset['label'], test_df_subset['prediction']))

    #bootstrap testing
    predictions = torch.tensor(test_df_subset['prediction'])
    labels = torch.tensor(test_df_subset['label'])
    results = bootstrap_testing(predictions, labels, metrics)
    print(results)



