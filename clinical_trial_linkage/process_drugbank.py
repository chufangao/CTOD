from tqdm import tqdm, trange
import xmltodict
import pandas as pd
import zipfile
import json


orange_book = pd.read_csv('./FDA_approvals/EOBZIP_2024_04/products.txt', sep='~').astype(str)

with zipfile.ZipFile('./drug-ndc-0001-of-0001.json.zip') as zf:
    with zf.open("drug-ndc-0001-of-0001.json") as f:
        data = json.load(f)
product = pd.DataFrame(data['results'])

with zipfile.ZipFile('./drugbank_all_full_database.xml.zip') as zf:
    with zf.open("full database.xml") as f:
        data_dict = xmltodict.parse(f.read())
data_dict = data_dict['drugbank']['drug']
print(data_dict[0].keys())


all_names = []
all_indications = []
all_products = []
all_ndc_product_code = []
all_ndc_id = []
all_synonyms = []
all_international_brands = []

for i in trange(len(data_dict)):
    drug = data_dict[i]

    # for k, v in drug.items(): # DO NOT PRINT FOR ALL DRUGS, WILL CRASH!!!
    #     print(k, v, '\n============================')
    # break
    all_names.append(drug['name'])
    all_indications.append(drug['indication'])

    #  ============ get synonyms ============
    if drug['synonyms'] is not None:
        if type(drug['synonyms']['synonym']) is not list:
            synonyms = [drug['synonyms']['synonym']]
        else:
            synonyms = drug['synonyms']['synonym']
        all_synonyms.append([_['#text'] for _ in synonyms])
    else:
        all_synonyms.append([])
    
    #  ============ get international brands ============
    if drug['international-brands'] is not None:
        if type(drug['international-brands']['international-brand']) is not list:
            international_brands = [drug['international-brands']['international-brand']]
        else:
            international_brands = drug['international-brands']['international-brand']
        all_international_brands.append([_['name'] for _ in international_brands])
    else:
        all_international_brands.append([])

    #  ============ get products ============
    if drug['products'] is not None:
        if type(drug['products']['product']) is not list:
            products = [drug['products']['product']]
        else:
            products = drug['products']['product']
        all_products.append([_['name'] for _ in products])
        all_ndc_id.append([_['ndc-id'] for _ in products])
        all_ndc_product_code.append([_['ndc-product-code'] for _ in products])
    else:
        all_products.append([])
        all_ndc_id.append([])
        all_ndc_product_code.append([])

drug_names = pd.DataFrame({'Name': all_names, 'Synonyms': all_synonyms, 'International Brands': all_international_brands, 
                            'Indications': all_indications, 'Products': all_products, 'Products NDC ID': all_ndc_id, 
                            'NDC Product Code': all_ndc_product_code})
drug_names.to_csv('processed_drug_names_all.csv', index=False)

# drug_names = pd.read_csv('drugbank/processed_drug_names.csv')
# drug_names['NDC Product Code'] = drug_names['NDC Product Code'].apply(lambda x: eval(x))
# drug_names['Products'] = drug_names['Products'].apply(lambda x: eval(x))

prod_ndc_dict = drug_names[['Products', 'NDC Product Code']].explode(['Products', 'NDC Product Code'])
prod_ndc_dict = pd.merge(prod_ndc_dict, product[['product_ndc','application_number']], left_on='NDC Product Code', right_on='product_ndc', how='left')
prod_ndc_dict = prod_ndc_dict.dropna(subset=['application_number'])

prod_ndc_dict['application_number'] = prod_ndc_dict['application_number'].str.replace(r'\D', '', regex=True)

prod_ndc_dict = prod_ndc_dict[prod_ndc_dict['application_number'].isin(orange_book['Appl_No'])]
prod_ndc_dict = prod_ndc_dict.set_index('NDC Product Code')['application_number'].to_dict()

drug_names['Appl_No'] = drug_names['NDC Product Code'].apply(lambda x: [prod_ndc_dict.get(i) for i in x]) 
# drop rows with no application number
drug_names = drug_names[drug_names['Appl_No'].apply(lambda x:  any([i is not None for i in x]))]
drug_names.to_csv('processed_drug_names_in_orangebook.csv', index=False)

# df = pd.read_csv('drugbank/processed_drug_names.csv', converters={'Synonyms': eval, 'International Brands': eval, 'Products': eval})
# assert 'Vivaglobin' in df['Products'].explode().unique() # True
