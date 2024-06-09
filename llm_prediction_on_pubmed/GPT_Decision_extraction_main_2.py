from support_functions import init_api
import pandas as pd
import json
import os
from tqdm import tqdm
from llama_index.llms.azure_openai import AzureOpenAI

model='gpt-35-turbo-16k'#"gpt-35-turbo-16k"
deployment_name='gpt-35-turbo-16k'#"gpt-35-turbo-16k"
temperature=0.2
api_key = '7a610cd3274042ba9480138c530c4025'
azure_endpoint = "https://surv.openai.azure.com/"
api_version = "2023-07-01-preview"
init_api(api_key, azure_endpoint, api_version)


data_path = '/home/jp65/CTOD/results/top_2_extracted_pubmed_articles.csv'
save_path = '/srv/local/data/jp65/gpt-35-decisions'#'/home/jp65/CTOD/results/gpt-35-decisions'

llm = AzureOpenAI(
    model=model,
    deployment_name=deployment_name,
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version,
)

from llama_index.core import Settings

Settings.llm = llm

# read csv file
pubmed_df = pd.read_csv(data_path)
print('length of pubmed_df:',len(pubmed_df))
# change NaN to empty string
pubmed_df = pubmed_df.fillna('')
pubmed_df.head()


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


gpt_decision = []
for i, row in tqdm(pubmed_df.iterrows(), total=pubmed_df.shape[0]):
    # try:
        nct_id = row['nct_id']
        official_title = row['official_title']
        save_name = f'{nct_id}_gpt_response.json'
        if save_name in os.listdir(save_path) or f'{nct_id}_gpt_response.txt' in os.listdir(save_path):
            # print(f'{nct_id} already exists')
            continue
        # top_1_similar_article_title = row['top_1_similar_article_title']
        # top_1_similar_article_abstract = row['top_1_similar_article_abstract']
        # top_1_similar_article_pub_date = row['top_1_similar_article_pub_date']
        # top_1_similar_article_type = row['top_1_similar_article_type']
        # top_1_similar_article_journal = row['top_1_similar_article_journal']
        
        
        # top_2_similar_article_title = row['top_2_similar_article_title']
        # top_2_similar_article_abstract = row['top_2_similar_article_abstract']
        # top_2_similar_pub_date = row['top_2_similar_article_pub_date']
        # top_2_similar_article_type = row['top_2_similar_article_type']
        # top_2_similar_article_journal = row['top_2_similar_article_journal']
        
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
            continue
    
    # print(response)
        # break
    # except Exception as e:
    #     print('Error in:', nct_id)
    #     # print(e)
    #     continue
