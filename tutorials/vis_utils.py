import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def plot_phase_distribution(studies_with_features,title):
  phase_group_list = ['EARLY_PHASE1','PHASE1','PHASE1/PHASE2','PHASE2','PHASE2/PHASE3','PHASE3','PHASE4']

  # get the count of trial in each phase
  phase_total_count = [studies_with_features[studies_with_features['phase'] == phase].shape[0] for phase in phase_group_list]

  phase_group_list = ['Early Phase 1','Phase 1','Phase 1/Phase 2','Phase 2','Phase 2/Phase 3','Phase 3','Phase 4']

  # Create the horizontal bar plot
  fig, ax = plt.subplots(figsize=(8, 6))
  bars = ax.barh(phase_group_list, phase_total_count, color=plt.cm.Reds(np.linspace(0.6, 0.5,  len(phase_group_list))), edgecolor='black',linewidth=2  ) 

  for bar in bars:
      width = bar.get_width()
      ax.annotate(f'{width}',
                  xy=(width, bar.get_y() + bar.get_height() / 2),
                  xytext=(3, 0),  # 3 points horizontal offset
                  textcoords="offset points",
                  ha='left', va='center',rotation=270,fontsize = 14)

  # Adding gridlines
  ax.xaxis.grid(True, linestyle='--', alpha=0.7)
  ax.yaxis.grid(False)
  ax.set_xlabel('Number of trials', fontsize=18, labelpad=3, weight='bold')
  ax.set_ylabel('Phase', fontsize=18, labelpad=3, weight='bold')
  ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
  ax.tick_params(axis='both', which='major', width=2, length=10)
  ax.tick_params(axis='x', which='minor', width=1, length=5)


  ax.set_title(title, fontsize=18, weight='bold', pad=18)
  ax.spines['bottom'].set_color('black')
  ax.spines['bottom'].set_linewidth(3)
  ax.spines['left'].set_color('black')
  ax.spines['left'].set_linewidth(3)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  ax.tick_params(axis='x', labelsize=18)
  ax.tick_params(axis='y', labelsize=18)
  plt.tight_layout()


def plot_completed_year_distribution(trial_data_df, title,thresh_timestamp='2025-01-01'):
  completion_time_df = trial_data_df[['nct_id','completion_date']]
  completion_time_df['completion_date'] = pd.to_datetime(completion_time_df['completion_date'], errors='coerce')
  completion_time_df = completion_time_df.dropna()

  # drop trials completed date after 2024 
  completion_time_df = completion_time_df[completion_time_df['completion_date'] < pd.Timestamp(thresh_timestamp)]
  completion_time_df = completion_time_df[completion_time_df['completion_date'] >= pd.Timestamp('1995-01-01')]

  # convert the completion date to year
  completion_time_df['completion_year'] = completion_time_df['completion_date'].dt.year
  completion_time_df

  # Yearly counts
  yearly_counts = completion_time_df['completion_year'].value_counts().sort_index()

  # Plot
  plt.figure(figsize=(12, 6))
  plt.plot(yearly_counts.index, yearly_counts.values, linestyle='-', marker='o', markersize=6,
          color='red', linewidth=2, label='Trials Completed')
  plt.title(title, fontsize=18, weight='bold', pad=18)
  plt.xlabel('Year', fontsize=18, labelpad=3,weight='bold')
  plt.ylabel('Number of Trials Completed', fontsize=18, labelpad=3,weight='bold')
  plt.xticks(ticks=yearly_counts.index, labels=yearly_counts.index, rotation=45, fontsize=16, ha='center')
  plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
  plt.yticks(fontsize=16)
  plt.tick_params(axis='both', which='major', width=2, length=10)
  plt.tick_params(axis='y', which='minor', width=1, length=5)
  plt.xlim(1994, 2025)


  plt.grid(axis='y', linestyle='--', alpha=0.7)
  ax = plt.gca()
  ax.spines['bottom'].set_color('black')
  ax.spines['bottom'].set_linewidth(3)
  ax.spines['left'].set_color('black')
  ax.spines['left'].set_linewidth(3)
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  plt.tight_layout()

def plot_top_K_condition_distribution(CTTI_PATH,trial_data_df, title,K=20):
  import zipfile
  with zipfile.ZipFile(CTTI_PATH, 'r') as zip_ref:
    names = zip_ref.namelist()
    condition_df = pd.read_csv(zip_ref.open([name for name in names if name.split("/")[-1]=='browse_conditions.txt'][0]), sep='|')
  condition_df = condition_df[['nct_id','mesh_type','downcase_mesh_term']]
  # filter only common nct_ids in condition_df and trial_data_df
  condition_df = condition_df[condition_df['nct_id'].isin(trial_data_df['nct_id'])]

  condition_df = condition_df[condition_df['mesh_type'] == 'mesh-ancestor']

  condition_dict = condition_df['downcase_mesh_term'].value_counts()
  condition_dict = condition_dict.to_dict()
  # get top 100 conditions
  condition_dict_2 = dict(sorted(condition_dict.items(), key=lambda item: item[1], reverse=True)[:K])

  # condition_dict = {k: v for k, v in condition_dict.items ()}
  condition_dict = {}
  for k, v in condition_dict_2.items():
    if k!= 'female urogenital diseases and pregnancy complications':
      condition_dict[k] =v
    else:
      condition_dict['fem. uro. dis. and preg. compl.'] = v


  # Extract categories and values
  categories = list(condition_dict.keys())
  values = list(condition_dict.values())
  # Create the bar plot
  plt.figure(figsize=(14, 10))
  plt.bar(categories, values, color=plt.cm.Reds(np.linspace(0.6, 0.5, len(categories))), alpha=1, edgecolor='black',linewidth=2)

  # Add labels and title
  plt.ylabel('Number of Trials', fontsize=18, labelpad=3, weight='bold')
  plt.xlabel('Conditions (MeSH Anscestors)', fontsize=18, labelpad=3, weight='bold')
  plt.xticks(rotation=90, fontsize=18)  
  plt.yticks(fontsize=18)
  plt.gca().margins(x=0.01) 
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.gca().spines['bottom'].set_color('black')
  plt.gca().spines['bottom'].set_linewidth(3)
  plt.gca().spines['left'].set_color('black')
  plt.gca().spines['left'].set_linewidth(3)
  plt.gca().spines['top'].set_visible(False)
  plt.gca().spines['right'].set_visible(False)
  plt.tick_params(axis='both', which='major', width=2, length=10)
  plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator())
  plt.tick_params(axis='y', which='minor', width=1, length=5)
  plt.tight_layout()

def drug_biologics_nct_ids_in_CTO(CTTI_PATH):
    import zipfile
    with zipfile.ZipFile(CTTI_PATH, 'r') as zip_ref:
      names = zip_ref.namelist()
      df = pd.read_csv(zip_ref.open([name for name in names if name.split("/")[-1]=='interventions.txt'][0]), sep='|')
      studies_df = pd.read_csv(zip_ref.open([name for name in names if name.split("/")[-1]=='studies.txt'][0]), sep='|')
    

    df = df[['nct_id','intervention_type']]
    type_list = ['drug','biological']
    df = df[df['intervention_type'].str.lower().isin(type_list)]
    
    # studies .txt
    
    studies_df = studies_df[studies_df['overall_status'].str.lower().isin(['terminated', 'withdrawn', 'suspended', 'withheld', 'no longer available', 'temporarily not available', 'approved for marketing', 'completed'])]
    studies_df.dropna(subset=['phase'], inplace=True)
    
    # get intersection of nct_ids
    df = df[df['nct_id'].isin(studies_df['nct_id'])]
    
    
    return df['nct_id'].tolist()