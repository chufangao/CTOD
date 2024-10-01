from token import OP
from turtle import up
from support_functions import init_api
import pandas as pd
import json
import os
from tqdm import tqdm
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
import argparse
from dotenv import load_dotenv
import numpy as np
import time



def main(top_2_pubmed_path,save_path,llm, upd_nct = None, dev = False):

    Settings.llm = llm

    # read csv file
    pubmed_df = pd.read_csv(top_2_pubmed_path)
    print('length of pubmed_df:',len(pubmed_df))
    # change NaN to empty string
    pubmed_df = pubmed_df.fillna('')


    prompt = '''
You are given the following PubMed abstracts for a clinical trial [NCT ID]. Your task is to use summarize important values of the trial into a json format. After summarization, you must predict the trial outcome. 

Guidelines:
- **Completeness**: Ensure there are no missing statistical tests and descriptions in JSON output.
- **Data Verification**: Before concluding the final answer, always verify that your observations align with the original trial description. Do not create any new information or hallucinate.

Additionally, generate a set of trial relevant questions and answers from the given abstracts. The questions should have two kinds of answers. One is a short descriptive sentence answering the question. The second is a set of five options, of which one is correct. 

Guidelines for the questions and answers:
- The questions shouldn't be on the trial outcome.
- Strictly use only the provided PubMed abstracts to generate questions and answers.
- Correctness is very important and do not hallucinate.
- Multiple answers cannot be correct. Only one of the options should be correct.

Output Format:
{
    "description": <string of text summary of the trial outcome>,
    "extracted features": [
        {
        "description: <string, text describing the feature extracted: e.g. "platelet response", "number of participants", "confidence interval", "p-value", "number", "study design">
        "value": <float or string of the values of above description>
        }, ... # can repeat as many times as needed
    ]
    "questions" : [
        {
            "question": <string, Description of the question>,
            "answer": <string, Description of the answer>,
            "options": [
                <optionA>,
                <optionB>,
                <optionC>,
                <optionD>,
                <optionE>
            ],
            "correct_option": <string, must be one of the options>
        }, ... # can repeat as many times as needed
    ],
    "outcome": <string, must either "success", "fail", "unsure">,
    "outcome reasoning": <string, reasoning as to why you predicted the outcome. Most trials succeed if the primary p-value < 0.05>
}

Notes for final output (Do not deviate from these instructions):
- Ensure the final answer format is a valid JSON dictionary.
- Strictly adhere to the JSON format and DO NOT include any additional text with the response.
- Ensure all "value" are of float or string ONLY  

Here is the title of the clinical trial. 

[CLINICAL TRIAL TITLE]

Here is the list of PubMed abstracts in the format of Metadata followed by Abstract: 

[ABSTRACT]

Begin!
'''

    # get gpt decision for each trial in the dataframe


    num = 0
    new_nctid_list = []
    upd_nctid_list = []
    for i, row in tqdm(pubmed_df.iterrows(), total=pubmed_df.shape[0]):
            nct_id = row['nct_id']
            # check if the nct_id is in the updated nct_ids
            if nct_id not in upd_nct:
                continue
            
            official_title = row['official_title']
            save_name = f'{nct_id}_gpt_response.json'
            if save_name in os.listdir(save_path) or f'{nct_id}_gpt_response.txt' in os.listdir(save_path):
                upd_nctid_list.append(nct_id)
            else:
                new_nctid_list.append(nct_id)
                # continue
            
            abstract_string = ''
            for k in range(1,3):
                if row[f'top_{k}_similar_article_title'] == '':
                    continue
                abstract_string += f'Reference type: {row[f"top_{k}_similar_article_type"]} \n\
        Title: {row[f"top_{k}_similar_article_title"]} \n\
        Journal: {row[f"top_{k}_similar_article_journal"]} \n\
        Date of Publication: {row[f"top_{k}_similar_article_pub_date"]} \n\
        Abstract: {row[f"top_{k}_similar_article_abstract"]} \n\n'
            
            edited_prompt = prompt.replace('[NCT ID]', nct_id)
            edited_prompt = edited_prompt.replace('[CLINICAL TRIAL TITLE]', official_title)
            edited_prompt = edited_prompt.replace('[ABSTRACT]', abstract_string)
            
            # print(edited_prompt)
            try:
                response = llm.complete(edited_prompt).text
            except:
                print('Error in:', nct_id)
                continue
            # print(response)
            try:
                response_dict = json.loads(response.strip())
            
                #save the response as a json file
                with open(os.path.join(save_path,f'{nct_id}_gpt_response.json'), 'w') as f:
                    json.dump(response_dict, f)
                    f.close()
            except:
                print('Error in:', nct_id)
                print('Saving as text file')
                # save as a text file
                with open(os.path.join(save_path,f'{nct_id}_gpt_response.txt'), 'w') as f:
                    f.write(response.strip())
                    f.close()
                # continue
            num += 1
            if dev and num == 2:
                print('Development mode: break')
                break
    
    # log all updated and new nct_id with date to log file
    log_path = '/'.join(save_path.split('/')[0:-2])
    with open(f'{log_path}/gpt_response_updated_new_nct_id.txt', 'a') as f:
        f.write('====================\n')
        f.write(f'Update time: {time.ctime()}\n')
        f.write('Following nct_ids GPT responses are updated:\n')
        f.write(f'Updated nct_ids: {upd_nctid_list} \n')
        f.write('Following nct_ids GPT responses are new:\n')
        f.write(f'New nct_ids: {new_nctid_list} \n')
        f.close()



if __name__ == '__main__':
    OPENAI_API_KEY= None # set your openai api key here
    load_dotenv()
    try:
        OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    except:
        pass
    
    temperature=0.2
    if OPENAI_API_KEY is None:
        raise ValueError('Please set the OPENAI_API_KEY')
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--top_2_pubmed_path', type=str, default= None, help='Path to the dataframe with top 2 extracted pubmed articles')
    parser.add_argument('--save_path', type=str, default= None, help='Path to save the LLM decisions')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    
    args = parser.parse_args()
    
    top_2_pubmed_path = os.path.join(args.save_path,'llm_predictions_on_pubmed','top_2_extracted_pubmed_articles.csv')
    
    
    save_path = os.path.join(args.save_path,'llm_predictions_on_pubmed','gpt_responses')
    
    # get updated nct_ids
    if os.path.exists(f'{args.save_path}/llm_predictions_on_pubmed/updated_new_nct_id.npy'):
        upd_nct = list(np.load(f'{args.save_path}/llm_predictions_on_pubmed/updated_new_nct_id.npy'))
    else:
        upd_nct = None
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    if top_2_pubmed_path is None:
        raise ValueError('Please provide the path to the dataframe with top 2 extracted pubmed articles')
    if save_path is None:
        raise ValueError('Please provide the path to save the LLM decisions')
    
    
    print('Initializing LLM')
    llm = OpenAI(model="gpt-3.5-turbo", temperature = temperature, api_key=OPENAI_API_KEY,
                 response_format = "json_object")

    
    
    main(top_2_pubmed_path,save_path,llm,upd_nct,args.dev)
    
    
