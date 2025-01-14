import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
import torch
import numpy as np
import random
import os
import copy
from transformers import AutoTokenizer, AutoModel


def set_seed(seed):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    
    # If you are using CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure that all operations are deterministic on GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for CUDA to ensure deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
    
def bootstrap_testing( y_prob, y_true, num_samples=100):
    y_pred = (y_prob > 0.5).astype(int)
    y_true = y_true.astype(int) 
    f1s = []
    aps = []
    rocs = []
    for _ in range(num_samples):
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        # convert to multiclass for precision recall curve
        y_true_multi = np.zeros((len(y_true), 2))
        y_true_multi[np.arange(len(y_true)), y_true] = 1
        y_pred_multi = np.zeros((len(y_pred), 2))
        y_pred_multi[np.arange(len(y_pred)), y_pred] = 1
        y_prob_multi = np.zeros((len(y_prob), 2))
        y_prob_multi[np.arange(len(y_prob)), y_pred] = y_prob
        
        # accs.append(np.mean(y_true[indices] == y_pred[indices]))
        f1s.append(f1_score(y_true_multi[indices], y_pred_multi[indices], average='weighted'))
        aps.append(average_precision_score(y_true_multi[indices], y_prob_multi[indices], average='weighted'))
        rocs.append(roc_auc_score(y_true_multi[indices], y_pred_multi[indices], average='weighted'))
    return np.mean(f1s), np.std(f1s), np.mean(aps), np.std(aps), np.mean(rocs), np.std(rocs)
    
# def bootstrap_testing(preds, target, threshold, bootstrap_num=20):
#     results = []
#     num_samples = len(target)
#     for _ in range(bootstrap_num):
#         cur_result = {}
#         idx = np.random.choice(num_samples, num_samples, replace=True)
#         cur_result["preds"] = preds[idx]
#         cur_result["target"] = target[idx]

#         # Apply the threshold for F1 score calculation
#         f1 = f1_score(cur_result["target"], (cur_result["preds"] >= threshold).astype(int))
#         roc_auc = roc_auc_score(cur_result["target"], cur_result["preds"])
#         pr_auc = average_precision_score(cur_result["target"], cur_result["preds"])

#         results.append({"F1": f1, "ROC-AUC": roc_auc, "PR-AUC": pr_auc})

#     result = {}
#     if len(results) == 1:
#         result = results[0]
#     elif len(results) > 1:
#         for key in results[0]:
#             data = [r[key] for r in results]
#             result[f"{key}_mean"] = np.mean(data)
#             result[f"{key}_std"] = np.std(data)
#     return result

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='dmis-lab/biobert-base-cased-v1.2', help='Model name') # microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
    args = parser.parse_args()
    
    short_model_name = args.model.split('/')[-1]
    # Example usage
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model)
    model = model.to(device)

    def get_bert_embeddings(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True).to(device)
        #if token length is greater than 512, create chunks of 512 tokens and get embeddings for each chunk and then average them
        if inputs['input_ids'].shape[1] > 512:
            embeddings = []
            for i in range(0, inputs['input_ids'].shape[1], 512):
                inputs_chunk = {k: v[:, i:i+512] for k, v in inputs.items()}
                outputs = model(**inputs_chunk)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy())
            return np.mean(embeddings, axis=0)
        else:
            outputs = model(**inputs)
            return outputs.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()
        
    train_data = pd.read_csv('../train_studies.csv')
    test_data = pd.read_csv('../test_studies.csv')

    train_data = test_data[(test_data['year']<2022)]
    test_data = test_data[(test_data['year']>=2022)]

    # # ======================== Obtain Embeddings ========================
    # print('train')
    # train_data['combined_embeddings'] = train_data['features'].apply(get_bert_embeddings)
    # print('valid')
    # # valid_data['combined_embeddings'] = valid_data['combined'].progress_apply(get_bert_embeddings)
    # valid_data = train_data
    # print('test')
    # test_data['combined_embeddings'] = test_data['features'].apply(get_bert_embeddings)

    # #save the embeddings
    # os.makedirs('data', exist_ok=True)
    # train_data.to_pickle(f'data/{short_model_name}_train_embeddings.pkl')
    # valid_data.to_pickle(f'data/{short_model_name}_valid_embeddings.pkl')
    # test_data.to_pickle(f'data/{short_model_name}_test_embeddings.pkl')

    #read the embeddings
    train_data = pd.read_pickle(f'data/{short_model_name}_train_embeddings.pkl')
    valid_data = pd.read_pickle(f'data/{short_model_name}_valid_embeddings.pkl')
    test_data = pd.read_pickle(f'data/{short_model_name}_test_embeddings.pkl')

    #drop duplicates
    train_data.drop_duplicates(subset='nct_id', inplace=True)
    valid_data.drop_duplicates(subset='nct_id', inplace=True)
    test_data.drop_duplicates(subset='nct_id', inplace=True)


    X_train = torch.tensor(np.stack(train_data['combined_embeddings'].values)).float().to(device)
    X_valid = torch.tensor(np.stack(valid_data['combined_embeddings'].values)).float().to(device)
    X_test = torch.tensor(np.stack(test_data['combined_embeddings'].values)).float().to(device)

    y_train = torch.tensor(train_data['label'].values).float().to(device)
    y_valid = torch.tensor(valid_data['label'].values).float().to(device)
    y_test = torch.tensor(test_data['label'].values).float().to(device)

    #NN model for predicting the column 'Trend' using the same X_train, y_train, X_test, y_test
    #train on the train data and validate on the validation data per 2 epochs to get the best model 
    nn_model= NN(input_dim=X_train.shape[1], hidden_dim=128, output_dim=1).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=1e-4)

    X_train_tensor = X_train
    y_train_tensor = y_train.view(-1, 1)

    X_valid_tensor = X_valid
    y_valid_tensor = y_valid.view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = TensorDataset(X_valid_tensor, y_valid_tensor)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    epochs = 100
    best_valid_loss = float('inf')
    best_model = copy.deepcopy(nn_model)

    # ======================== Training Loop ========================
    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        nn_model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = nn_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        nn_model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                output = nn_model(data)
                loss = criterion(output, target)
                valid_loss += loss.item()
        
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model = copy.deepcopy(nn_model)
            # torch.save(nn_model.state_dict(), 'nn_model.pt')

    # ======================== Prediction Step ========================
    
    #load the best model and predict on the test data
    # nn_model.load_state_dict(torch.load('nn_model.pt'))
    nn_model = best_model
    nn_model.eval()

    test_loader = DataLoader(TensorDataset(X_test, y_test.view(-1, 1)), batch_size=32, shuffle=False)

    y_pred = []
    y_true = []

    with torch.no_grad():
        for data, target in test_loader:
            output = nn_model(data)
            y_pred.extend(output.cpu().numpy())
            y_true.extend(target.cpu().numpy())

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    test_data['pred'] = y_pred

    # ======================== Calculate Metrics ========================
    threshold = 0.5
    # Loop through each phase and calculate metrics
    for phase in ['1', '2', '3']:
        test_df_subset = test_data[test_data['phase'].str.lower().str.contains(phase)]

        # print(phase)
        # f1 = f1_score(test_df_subset['label'], (test_df_subset['pred'] >= threshold).astype(int))
        # pr_auc = average_precision_score(test_df_subset['label'], test_df_subset['pred'])
        # roc_auc = roc_auc_score(test_df_subset['label'], test_df_subset['pred'])
        # print(f"F1: {f1}, PR-AUC: {pr_auc}, ROC-AUC: {roc_auc}")
        
        # Bootstrap testing
        target = test_df_subset['label'].values
        preds = test_df_subset['pred'].values
        
        f1_mean, f1_std, ap_mean, ap_std, roc_mean, roc_std = bootstrap_testing(preds, target)
        print(f"{phase}, {f1_mean:.3f}, {f1_std:.3f}, {ap_mean:.3f}, {ap_std:.3f}, {roc_mean:.3f}, {roc_std:.3f}")

    # save the predictions as csv
    test_data.to_csv(f'data/{short_model_name}_test_predictions.csv', index=False)