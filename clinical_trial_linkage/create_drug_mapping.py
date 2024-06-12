from regex import F
from tqdm import tqdm
import os
import pandas as pd
import re
import json


def remove_superscripts(text):
    # Define a regular expression pattern to match superscripted characters
    pattern = r"[\u00B2\u00B3\u00B9\u2070\u2074\u2075\u2076\u2077\u2078\u2079\u207A\u207B\u207C\u207D\u207E\u207F]"
    
    # Use the re.sub() function to replace the matched superscripts with an empty string
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text

def remove_special_chars(text):
    # Define a regular expression pattern to match superscripted characters and registered trademark symbol
    pattern = r"[\u00B2\u00B3\u00B9\u2070\u2074\u2075\u2076\u2077\u2078\u2079\u207A\u207B\u207C\u207D\u207E\u207F\u00AE]"
    
    # Use the re.sub() function to replace the matched characters with an empty string
    cleaned_text = re.sub(pattern, "", text)
    
    return cleaned_text

def clean_string(text):
    # Remove superscripts
    text = remove_superscripts(text)
    
    # Remove special characters
    text = remove_special_chars(text)
    
    return text

def main(drug_bank_process_csv_path,save_path):
    df = pd.read_csv(drug_bank_process_csv_path,
                    converters={'Synonyms': eval, 'International Brands': eval, 'Products': eval})

    #create a mapping dict to map synonyms and product names to drug generic names
    drug_mapping = {}
    for i in tqdm(range(len(df))):
        drug_mapping[clean_string(df['Name'][i]).lower()] = [clean_string(df['Name'][i]).lower()]
        for syn in df['Synonyms'][i]:
            if clean_string(syn.lower()) in drug_mapping:
                drug_mapping[clean_string(syn.lower())].append(clean_string(df['Name'][i]).lower().strip())
            else:
                drug_mapping[clean_string(syn.lower())] = [clean_string(df['Name'][i]).lower().strip()]
        for product in list(df['Products'][i]):
            if clean_string(product.lower()) in drug_mapping:
                drug_mapping[clean_string(product.lower())].append(clean_string(df['Name'][i]).lower().strip())
            else:
                drug_mapping[clean_string(product.lower())] = [clean_string(df['Name'][i]).lower().strip()]
        # break
        
    print(f'Number of drugs in drug mapping: {len(drug_mapping)}')

    # go through the drug mapping dict and delete duplicates 
    for key in drug_mapping:
        drug_mapping[key] = list(set(drug_mapping[key]))
    print(f'Number of drugs in drug mapping after removing duplicates: {len(drug_mapping)}')
    
    with open(os.path.join(save_path, 'drug_mapping.json'), 'w') as f:
        json.dump(drug_mapping, f)
        
        
if __name__ == '__main__':
    drug_bank_process_csv_path = './drugbank/processed_drug_names_all.csv'
    save_path = './'
    main(drug_bank_process_csv_path,save_path)