# TOP baselines
import zipfile
import pandas as pd
import numpy as np
import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, f1_score, average_precision_score, roc_auc_score

def bootstrap_eval(y_true, y_pred, y_prob, num_samples=100):
    f1s = []
    aps = []
    rocs = []
    for _ in range(num_samples):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        f1s.append(f1_score(y_true[indices], y_pred[indices]))
        aps.append(average_precision_score(y_true[indices], y_prob[indices]))
        rocs.append(roc_auc_score(y_true[indices], y_prob[indices]))
    return np.mean(f1s), np.std(f1s), np.mean(aps), np.std(aps), np.mean(rocs), np.std(rocs)

# run = 'comparison_with_TOP'
# run = 'pre2020'
# run = 'post2020'

# for run in ['TOP','comparison_with_TOP', 'pre2020', 'post2020']:
train_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*train.csv"))])
valid_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*valid.csv"))])
test_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*test.csv"))])
for run in ['comparison_with_TOP',]:
# for run in ['post2020']:
    if run == 'TOP':
        path = './clinical-trial-outcome-prediction/data/'
        train_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*train.csv"))])
        valid_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*valid.csv"))])
        test_df = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(path, "phase*test.csv"))])
    elif run == 'comparison_with_TOP':
        with zipfile.ZipFile('./test_train_spit_comparison_with_TOP.zip') as zf:
            train_df = pd.read_csv(zf.open('test_train_spit_comparison_with_TOP/CTOD_train_2014.csv'))
            valid_df = pd.read_csv(zf.open('test_train_spit_comparison_with_TOP/CTOD_valid_2014.csv'))
            test_df = pd.read_csv(zf.open('test_train_spit_comparison_with_TOP/TOP_test.csv'))
    elif run == 'pre2020':
        with zipfile.ZipFile('./CTOD_splits.zip') as zf:
            train_df = pd.read_csv(zf.open('CTOD_splits/train_pre2020.csv'))
            valid_df = pd.read_csv(zf.open('CTOD_splits/valid_pre2020.csv'))
            test_df = pd.read_csv(zf.open('CTOD_splits/test_pre2020.csv'))
    elif run == 'post2020':
        with zipfile.ZipFile('./CTOD_splits.zip') as zf:
            train_df = pd.read_csv(zf.open('CTOD_splits/train_post2020.csv'))
            valid_df = pd.read_csv(zf.open('CTOD_splits/valid_post2020.csv'))
            test_df = pd.read_csv(zf.open('CTOD_splits/test_post2020.csv'))

    print(train_df.shape, valid_df.shape, test_df.shape)
    print('Train', np.unique(train_df['label'], return_counts=True))
    print('Valid', np.unique(valid_df['label'], return_counts=True))
    print('Test', np.unique(test_df['label'], return_counts=True))

    train_df = pd.concat([train_df, valid_df])
    train_df.fillna('', inplace=True)
    train_df.drop_duplicates(subset=['nctid'], inplace=True)
    test_df.fillna('', inplace=True)
    test_df.drop_duplicates(subset=['nctid'], inplace=True)
    # train_df['features'] = train_df['status'] + ' ' + train_df['why_stop'] + ' ' + train_df['diseases'] + ' ' + train_df['drugs'] + ' ' + train_df['criteria']
    # test_df['features'] = test_df['status'] + ' ' + test_df['why_stop'] + ' ' + test_df['diseases'] + ' ' + test_df['drugs'] + ' ' + test_df['criteria']
    # train_df['features'] = train_df['phase'] + ' '  + train_df['diseases'] + ' '  + train_df['icdcodes'] + ' ' + train_df['drugs'] + ' ' + train_df['criteria']
    # test_df['features'] = test_df['phase'] + ' '  + test_df['diseases'] + ' '  + test_df['icdcodes'] + ' ' + test_df['drugs'] + ' ' + test_df['criteria']
    train_df['features'] = train_df['phase'] + ' '  + train_df['diseases'] + ' '  + train_df['icdcodes'] + ' ' + train_df['drugs'] + ' ' + train_df['criteria']
    test_df['features'] = test_df['phase'] + ' '  + test_df['diseases'] + ' '  + test_df['icdcodes'] + ' ' + test_df['drugs'] + ' ' + test_df['criteria']

    # tfidf = TfidfVectorizer(max_features=2048, stop_words='english')
    tfidf = TfidfVectorizer(max_features=2048, stop_words='english')
    X_train = tfidf.fit_transform(train_df['features'])
    X_test = tfidf.transform(test_df['features'])

    print(f'Model, Phase, F1, AP, ROC')
    for model_name in ['svm', 'xgboost', 'mlp', 'rf', 'lr', ]:
    # for model_name in ['svm', 'lr']:
    # for model_name in ['svm']:
        if model_name == 'rf':
            model = RandomForestClassifier(n_estimators=300, random_state=0, max_depth=10, n_jobs=4)
        elif model_name == 'lr':
            model = LogisticRegression(max_iter=1000, random_state=0)
        elif model_name == 'svm':
            model = LinearSVC(dual="auto", max_iter=10000, random_state=0)
            model = CalibratedClassifierCV(model) 
            # model = SVC(kernel='linear', probability=True, random_state=0)
        elif model_name == 'xgboost':
            model = XGBClassifier(n_estimators=300, random_state=0, max_depth=10, n_jobs=4)
        elif model_name == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=0)
        else:
            raise ValueError('Unknown model name')

        model.fit(X_train, train_df['label'])
        test_df['pred'] = model.predict(X_test)
        test_df['prob'] = model.predict_proba(X_test)[:, 1]
        # print(test_df['pred'])

        for phase in ['phase 1', 'phase 2', 'phase 3']:
            test_df_subset = test_df[test_df['phase'].str.lower().str.contains(phase)]
            # print(phase, test_df_subset.shape)
            # print(classification_report(test_df_subset['label'], test_df_subset['pred']))
            f1_mean, f1_std, ap_mean, ap_std, roc_mean, roc_std = bootstrap_eval(test_df_subset['label'].values, test_df_subset['pred'].values, test_df_subset['prob'].values)
            print(f"{phase}, {model_name}, {f1_mean:.3f}, {f1_std:.3f}, {ap_mean:.3f}, {ap_std:.3f}, {roc_mean:.3f}, {roc_std:.3f}")
