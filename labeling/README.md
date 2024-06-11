# Labeling CTO

## Prerequisites

- Please ensure that the paths are correct in llm_qa.py, specifically, the path to the CTTI datasets as well as to the other directories
- Clone the directory for the publicly available [TOP](https://www.sciencedirect.com/science/article/pii/S2666389922000186) dataset.
```jsx
git clone https://github.com/futianfan/clinical-trial-outcome-prediction
```
- Obtain the pipe delimited files from CTTI here: https://aact.ctti-clinicaltrials.org/download 
- **Note:** that we also provide our copy of the CTTI in our Zenodo uploaded files for reproducibility.

## Label Creation
- Please look at the [create_labels.ipynb](https://github.com/chufangao/CTOD/blob/main/labeling/create_labels.ipynb) to see how our label creation works. This should work out of the box after all python packages are installed and all supplemetary files downloaded from Zenodo.

## Label Splitting
- Additionally, please see our pre-split labels for Pre and Post 2020 trial outcomes, segmented by Data Programming and Random Forest label creation methods respectively.