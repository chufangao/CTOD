import argparse
import os



if __name__ == '__main__':
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
    
    
        
        