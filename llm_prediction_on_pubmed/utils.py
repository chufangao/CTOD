import pandas as pd 

def drug_biologics_nct_ids(intervention_path):
    df = pd.read_csv(intervention_path, sep='|')
    df = df[['nct_id','intervention_type']]
    type_list = ['drug','biological']
    df = df[df['intervention_type'].str.lower().isin(type_list)]
    
    return df['nct_id'].tolist()
