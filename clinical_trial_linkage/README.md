# Clinical Trial Linkage Generation

![trial_linkage_algorithm.png](trial_linkage_algorithm.png)

## Prerequisites

- Download the [FDA orange book](https://www.fda.gov/media/76860/download?attachment) and save it to ./FDA_approvals/.  Currently, we provide the downloaded version as of 2024-04, which was used to create our dataset. Refer [https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files](https://www.fda.gov/drugs/drug-approvals-and-databases/orange-book-data-files) for additional details on the FDA orange book.
- Download the trial dataset from CITI <path>. If it has already been downloaded, provide the path to the data in the scripts.
- Download the National Drug Code Directory [/drug/ndc](https://open.fda.gov/data/downloads/)
- Download the DrugBank full database for researchers: https://go.drugbank.com/releases/latest
- We have provided the drug mapping dictionary to map the drug names to their generic names at drug_mapping.json. Additionally, to reproduce the mapping dictionary and extract from the updated drug bank database, run the following:
```jsx

python process_drugbank.py
python create_drug_mapping.py
```

## 1. Extract trial info and save trial embeddings

First, we extract trial features from the CITI dataset. Provide the <data_path> for downloaded CITI data in the command below:

```jsx
cd clinical_trial_linkage
python extract_trial_info.py --data_path < Path to data files folder from CITI >
```

Run the following command to extract and save the embeddings for the trial features using PubMedBERT. Make sure to provide the path to save the embeddings. Feel free to make changes to num_workers and gpu_ids as necessary.

```jsx
python get_embedding_for_trial_linkage.py --root_folder < Path to save the embeddings > --num_workers 2 --gpu_ids 0,1
```

## 2. Linking of clinical trials across phases

Based on the extracted embeddings, we link the trials across different phases, as shown in the above figure. Run the following command to link the trials across phases. Provide the root_folder path to save the extracted linkages and embedding_path with the saved embeddings. Since we link from the latter phase to the initial phases, provide the starting later phase to target_phase. Also, we need only to consider the following phases to create the trial linkage: ['Phase 2', 'Phase 2/Phase 3', 'Phase 3', 'Phase 4']

```jsx
# Phase 4
python create_trial_linkage.py --root_folder < Folder to save the created linkages > --target_phase 'Phase 4' --embedding_path < Folder containing the embeddings saved for the trials > --num_workers 2 --gpu_ids 0,1

# Phase 3
python create_trial_linkage.py --root_folder < Folder to save the created linkages > --target_phase 'Phase 3' --embedding_path < Folder containing the embeddings saved for the trials > --num_workers 2 --gpu_ids 0,1

# Phase 2/Phase 3
python create_trial_linkage.py --root_folder < Folder to save the created linkages > --target_phase 'Phase 2/Phase 3' --embedding_path < Folder containing the embeddings saved for the trials > --num_workers 2 --gpu_ids 0,1

# Phase 2
python create_trial_linkage.py --root_folder < Folder to save the created linkages > --target_phase 'Phase 2' --embedding_path < Folder containing the embeddings saved for the trials > --num_workers 2 --gpu_ids 0,1
```

## 3. Extract outcome labels

Run the following command to extract clinical trial outcome weak labels from clinical trial linkages. Provide the path with saved trial linkages.

```jsx
python extract_outcome_from_trial_linkage.py --trial_linkage_path <Path to the trial linkage folder containing the JSON files of the trial linkage>
```

## 4. FDA approval matching

Run the following command to match the FDA approvals from the orange book to phase 3 trials and update the outcome labels for phase 3 trials.

```jsx
python match_fda_approvals.py --trial_linkage_path <Path to save the matched trials results (provide the path where the trial linkages results are saved)> 
```

The final outcome labels from the clinical trial linkages and matching FDA approvals will be saved at 

```jsx
<trial_linkage_path>/outcome_labels/Merged_(ALL)_trial_linkage_outcome_df.csv
```
