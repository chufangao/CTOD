"""
Label Arrangement Script

This script organizes and copies generated labels from different CTOD modules 
into a centralized outcome_labels directory for easy access and analysis.

The script consolidates:
- GPT-based predictions from LLM module
- Trial linkage outcomes from clinical trial linkage module
- Other weak supervision labels

Usage:
    python arrange_labels.py --save_path <SAVE_PATH>

Args:
    --save_path: Base directory containing module outputs

Output:
    Creates outcome_labels/ directory with consolidated label files
"""

import argparse
import os



if __name__ == '__main__':
    """
    Main execution function for label arrangement.
    
    This script consolidates label outputs from different CTOD modules:
    1. Copies GPT-based predictions from LLM module
    2. Copies trial linkage outcomes from clinical trial linkage module
    3. Organizes all labels in a centralized outcome_labels/ directory
    
    The organized labels can then be used for model training and evaluation.
    """
    parser = argparse.ArgumentParser(description='Arrange labels for CTOD dataset')
    parser.add_argument('--save_path', type=str, help='Path to the folder to save the arranged labels')

    args = parser.parse_args()
    
    label_save_path = os.path.join(args.save_path, 'outcome_labels')
    if not os.path.exists(label_save_path):
        os.makedirs(label_save_path)
    #copy gpt labels to label_save_path
    gpt_labels_path = os.path.join(args.save_path, 'llm_predictions_on_pubmed/pubmed_gpt_outcomes.csv')
    
    os.system(f'cp {gpt_labels_path} {label_save_path}/pubmed_gpt_outcomes.csv')
    
    # copy trial linkage labels to label_save_path
    trial_linkage_labels_path = os.path.join(args.save_path, 'clinical_trial_linkage/trial_linkages/outcome_labels/Merged_all_trial_linkage_outcome_df__FDA_updated.csv ')
    trial_linkage_save_path = str(os.path.join(label_save_path, 'Merged_all_trial_linkage_outcome_df__FDA_updated.csv'))
    os.system(f"cp {trial_linkage_labels_path} {trial_linkage_save_path}")
    
    
        
        