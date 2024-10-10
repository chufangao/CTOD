import gradio as gr
import pandas as pd
import glob
from rapidfuzz import process, fuzz
import os
import numpy as np
import json

def process_label_preds(df):
    cols_to_drop = ['phase_x','phase_y','hint','hint.1','hint.2']
    df.drop(cols_to_drop, axis=1, inplace=True)
    columns = list(df.columns)
    inds_to_keep = []
    current_columns = []
    for i, col in enumerate(columns):
        if col not in current_columns:
            current_columns.append(col)
            inds_to_keep.append(i)
    df = df.iloc[:, inds_to_keep]
    # convert all columns not named nct_id to int
    for col in df.columns:
        if col != 'nct_id':
            df[col] = df[col].astype(int)
            # if row == 0, replace with "Failure", -1 wtih Abstain, and 1 with "Success"
            df[col] = df[col].replace(0, "Failure")
            df[col] = df[col].replace(-1, "Abstain")
            df[col] = df[col].replace(1, "Success")
            
    return df

# # we always test on supervised TOP labels
# train_df = pd.read_csv('../labeling/pre_post_2020/train_pre2020_dp.csv')
# valid_df = pd.read_csv('../labeling/pre_post_2020/valid_pre2020_dp.csv')
# test_df = pd.read_csv('../labeling/pre_post_2020/test_pre2020_dp.csv')
# all_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
all_df = pd.read_csv('../CTTI/studies.txt', sep='|')
all_df = all_df[['nct_id', 'brief_title', 'official_title',  'overall_status', 'phase', 'enrollment', 'why_stopped', 'study_type',  'start_date','completion_date']]

phase_1_label_preds = pd.read_csv('../labeling/weak_preds_by_phase/phase1_dp.csv')
phase_2_label_preds = pd.read_csv('../labeling/weak_preds_by_phase/phase1_dp.csv')
phase_3_label_preds = pd.read_csv('../labeling/weak_preds_by_phase/phase1_dp.csv')
phase_1_label_preds = process_label_preds(phase_1_label_preds)
phase_2_label_preds = process_label_preds(phase_2_label_preds)
phase_3_label_preds = process_label_preds(phase_3_label_preds)

all_nct_ids = pd.concat([phase_1_label_preds['nct_id'], phase_2_label_preds['nct_id'], phase_3_label_preds['nct_id']]).values
all_nct_ids = set(all_nct_ids)
all_df = all_df[all_df['nct_id'].isin(all_nct_ids)]

print(all_df.shape)
all_brief_titles = list(all_df['brief_title'].values)

linkage_path = "/srv/local/data/chufan2/github/CTOD/supplementary/clinical_trial_linkage/Merged_(ALL)_trial_linkage_outcome_df_FDA_updated.csv"
linkage_df = pd.read_csv(linkage_path)

gpt_decision_path = '../supplementary/llm_prediction_on_pubmed/gpt-35-decisions/'

def get_gpt_decisions(nct_id):
    nct_id = nct_id.strip()
    if os.path.exists(gpt_decision_path + nct_id + '_gpt_response.json'):
        try:
            with open(gpt_decision_path + nct_id + '_gpt_response.json', 'r') as f:
                json_dict = json.loads(f.read())
        except json.JSONDecodeError:
            return {}
        return json.dumps(json_dict, indent=4)

def get_closest_nctids(title, n=5):
    title = title.strip()
    if title.startswith('NCT'):
        return all_df.loc[all_df['nct_id'] == title]
    # fuzzy string match brief_title
    # print(title)
    closest_titles = process.extract(title, all_brief_titles, scorer=fuzz.WRatio, limit=n)
    # print(closest_titles)
    closest_inds = [_[2] for _ in closest_titles] # only return the titles
    return all_df.iloc[closest_inds,:]

def get_lf_preds(nct_id):
    nct_id = nct_id.strip()
    # nct_id = value['nct_id']
    # get the label predictions for the nct_id
    # if phase 1
    if nct_id not in all_df['nct_id'].values:
        return None
    phase = all_df.loc[all_df['nct_id'] == nct_id, 'phase'].values[0]
    if 'Phase 1' in phase:
        ret = phase_1_label_preds.loc[phase_1_label_preds['nct_id'] == nct_id]
    elif 'Phase 2' in phase:
        ret = phase_2_label_preds.loc[phase_2_label_preds['nct_id'] == nct_id]
    elif 'Phase 3' in phase:
        ret = phase_3_label_preds.loc[phase_3_label_preds['nct_id'] == nct_id]
    else:
        ret = None
    print(ret)
    return ret

def get_lf_linkages(nct_id):
    nct_id = nct_id.strip()
    if nct_id not in linkage_df['nctid'].values:
        return None
    ret = linkage_df.loc[linkage_df['nctid'] == nct_id]
    return ret

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    input_title = gr.Textbox(label="Trial Title Search")
    output = gr.DataFrame(label="Trial Search Results", wrap=True)
    # greet_btn = gr.Button("Search")
    input_nctid = gr.Textbox(label="View Specific NCT ID")
    output2 = gr.DataFrame(label="Weakly Supervised Label Predictions", wrap=True)
    output3 = gr.DataFrame(label="Next Phase Link Prediction", wrap=True)
    output4 = gr.Textbox(label="GPT Decisions",)

    gr.on(
        triggers=[input_title.submit],
        fn=get_closest_nctids,
        inputs=input_title,
        outputs=output,
    )

    gr.on(
        triggers=[input_nctid.submit],
        fn=get_lf_preds,
        inputs=input_nctid,
        outputs=output2,
    )
    gr.on(
        triggers=[input_nctid.submit],
        fn=get_lf_linkages,
        inputs=input_nctid,
        outputs=output3,
    )
    gr.on(
        triggers=[input_nctid.submit],
        fn=get_gpt_decisions,
        inputs=input_nctid,
        outputs=output4,
    )

    # output.select(fn=get_lf_preds, inputs=output, outputs=output2)

    # also if enter key is pressed

    # @gr.render(inputs=output)
    # def show_split(text):
    #     if len(text) == 0:
    #         gr.Markdown("## No Input Provided")
    #     else:
    #         for letter in text:
    #             gr.Textbox(letter)

demo.launch(share=True)
