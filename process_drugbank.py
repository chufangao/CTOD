from tqdm import tqdm, trange
import xmltodict
import pandas as pd
from zipfile import ZipFile

# # drug_vocab = pd.read_csv('./drugbank/drugbank vocabulary.csv')
# drugbank_approved = pd.read_csv('./drugbank/drugbank_all_drug_links.csv.zip')

# # list(all_synonyms).index('vivaglobin') # not found
# # list(common_names).index('vivaglobin') # not found

with ZipFile('/srv/local/data/chufan2/clinical_trial_data_test/drugbank/drugbank_all_full_database.xml.zip') as zf:
    with zf.open("full database.xml") as f:
        data_dict = xmltodict.parse(f.read())
        data_dict = data_dict['drugbank']['drug']
print(data_dict[0].keys())

all_names = []
all_products = []
all_synonyms = []
all_international_brands = []

for i in trange(len(data_dict)):
    drug = data_dict[i]
    name = drug['name']
    all_names.append(name)

    if drug['synonyms'] is not None:
        if type(drug['synonyms']['synonym']) is not list:
            synonyms = [drug['synonyms']['synonym']]
        else:
            synonyms = drug['synonyms']['synonym']
        synonyms = [_['#text'] for _ in synonyms]
        all_synonyms.append(synonyms)
    else:
        all_synonyms.append([])
    
    if drug['international-brands'] is not None:
        if type(drug['international-brands']['international-brand']) is not list:
            international_brands = [drug['international-brands']['international-brand']]
        else:
            international_brands = drug['international-brands']['international-brand']
        international_brands = [_['name'] for _ in international_brands]
        all_international_brands.append(international_brands)
    else:
        all_international_brands.append([])

    if drug['products'] is not None:
        if type(drug['products']['product']) is not list:
            products = [drug['products']['product']]
        else:
            products = drug['products']['product']
        products = [_['name'] for _ in products]
        all_products.append(products)
    else:
        all_products.append([])

df = pd.DataFrame({'Name': all_names, 'Synonyms': all_synonyms, 'International Brands': all_international_brands, 'Products': all_products})
df.to_csv('./drugbank/processed_drug_names.csv', index=False)

df = pd.read_csv('/srv/local/data/chufan2/clinical_trial_data_test/drugbank/processed_drug_names.csv', converters={'Synonyms': eval, 'International Brands': eval, 'Products': eval})
assert 'Vivaglobin' in df['Products'].explode().unique() # True