import inDelphi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import pickle

from pandas.plotting import table 
from scipy.stats import gaussian_kde
from mylib import util

# Render df as a table to generate figure
def render_mpl_table(df, col_width):
    plt.rcParams["font.family"] = "monospace"

    size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([col_width, 0.7])
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')
    
    mpl_table = ax.table(cellText=df.values, bbox=[0, 0, 1, 1], colLabels=df.columns)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(16)

    mpl_table.auto_set_column_width(col=list(range(len(df.columns))))

    return ax.get_figure()

# Adds an indel column (used to find all repair outcome observation % of a particle sequence + outcome),
# and add the observed repair outcome frequency to each row
def add_indel_column(pred_df, stats, observed_freqs):
  new_pred_df = pred_df
  indels = []
  freqs = []
  if type(stats) == dict:
    seq = stats['Reference sequence']
    cutsite = stats['Cutsite']
  else:
    seq = stats['Reference sequence'].iloc[0]
    cutsite = stats['Cutsite'].iloc[0]

  # For each row in the df (repair outcomes of the sequence), create an indel column and find the observed frequency, from the freqs dict
  for idx, row in new_pred_df.iterrows():
    gt_pos = row['Genotype position']

    if row['Category']  == 'ins':
        indel = '1+' + str(row['Inserted Bases'])
    else:
      gt_pos = int(gt_pos)
      dl = row['Length']
      indel = str(-dl + gt_pos) + '+' + str(dl)
    
    # Not sure why this is necessary, might be because of predicted indels that were not observed?
    if indel in observed_freqs[seq]:   
        freqs.append(observed_freqs[seq][indel])
    else:
       freqs.append(0)

    indels.append(indel)
  
  # Add indels and observed frequences to the df and return
  new_pred_df['indel'] = indels
  new_pred_df['Obs.%'] = freqs
  return new_pred_df

# add the | at the place of the cutsite and . in the place of deleted nucleotides to each sequence
def set_genotype(pred_df, stats):
  new_pred_df = pred_df
  exps = []

  if type(stats) == dict:
    seq = stats['Reference sequence']
    cutsite = stats['Cutsite']
  else:
    seq = stats['Reference sequence'].iloc[0]
    cutsite = stats['Cutsite'].iloc[0]

  # For each row in the df (repair outcomes of the sequence), change the genotype to add | and . in the case of a deletion
  for idx, row in new_pred_df.iterrows():
    gt_pos = row['Genotype position']
    if row['Category']  == 'ins':
        exp = row['Genotype'][:cutsite] + row['Genotype'][cutsite:]
    else:
        gt_pos = int(gt_pos)
        dl = row['Length']
        exp = seq[:cutsite - dl + gt_pos] + "." * abs(-dl + gt_pos)  + "|" + "." * abs(gt_pos) + seq[cutsite + gt_pos:]
    exps.append(exp)

  # Update genotype column in the df and return
  new_pred_df['Genotype'] = exps
  return new_pred_df

# Generate figure 1e
def generate_figure_1e(test_sequences, cutsite, observed_freqs):
    pd.set_option('display.max_colwidth', 199)

    total_df = []

    for sequence in test_sequences:
        # Get predictions and statistics
        pred_df, stats = inDelphi.predict(sequence, cutsite)

        # Needed because of python version
        pred_df = pred_df.rename(columns={'Genotype position': 'Genotype_position'})

        # Correctly identify insertions
        insQuery = '(Category == \'ins\')'
        pred_df.loc[pred_df.query(insQuery).index,'Cat'] = "ins"

        # Correctly identify mh deletions
        mhQuery = '(Category == \'del\') & (Genotype_position != \'e\')'
        pred_df.loc[pred_df.query(mhQuery).index,'Cat'] = "mh del"

        # Needed because of python version
        pred_df = pred_df.rename(columns={'Genotype_position': 'Genotype position'})

        # Add mhless_genotypes
        pred_df = inDelphi.add_mhless_genotypes(pred_df, stats)

        # Add a genotype column
        pred_df = inDelphi.add_genotype_column(pred_df, stats)

   
        # Set cut line and add inserted nucleotides to genotype
        pred_df = set_genotype(pred_df, stats)

        # Add indel column & observed frequencies
        pred_df = add_indel_column(pred_df, stats, observed_freqs)
 
        print(pred_df.sort_values(by='Predicted frequency', ascending=False).head(6))
        
        pred_df = pred_df.rename(columns={'Predicted frequency': 'Pred.%'})
      
        # Get the top 6 most frequent repair outcomes
        sorted_df = pred_df.sort_values(by='Obs.%', ascending=False).head(6)[['Genotype', 'Cat', 'Obs.%', 'Pred.%']]

        # Round all columns to 1 decimal
        sorted_df = sorted_df.round(1)

        # Add the sequence we are editing on to the top of teh df
        sorted_df.iloc[0] = [sequence[:stats['Cutsite']] + '|' + sequence[stats['Cutsite']:], '', '', ''] 

        # Add result of this sequence to dataframe used to generate the figure
        total_df.append(sorted_df)

    pd.reset_option('display.max_rows')
    fig = render_mpl_table(pd.concat(total_df), col_width=4.0)
    fig.savefig("figures/figure_1e.png",dpi=300, bbox_inches = "tight")


def generate_figure_3f(test_targets):
    #           GTTCTACAGATTGCTTG|TAC
    # AACACTCCCTGTTCTACAGATTGCTTG|TACTGGTGAACAACATATTGGTATTCAA
    # iterate over all test targets
    highest_del_freq_list = []
    highest_ins_freq_list = []
    for target in test_targets.values():
        _, stats = inDelphi.predict(target, 27)
        highest_del_freq_list.append(stats["Highest del frequency"])
        highest_ins_freq_list.append(stats["Highest ins frequency"])

    # Make a histogram of the frequencies
    # kde = gaussian_kde(highest_del_freq_list)
    fig, ax = plt.subplots(1, 1)
    # x = np.linspace(0, 100, 1000)
    ax.hist(highest_del_freq_list, bins = 100, density=False, color = 'red', alpha = 0.5)
    # ax.plot(x, kde(x), linewidth=2, color='c')
    # Adds labels to the figure
    ax.set_xlabel('Most frequent deletion genotype (%)')
    ax.set_ylabel('Number of gRNAs')

    # Save figure to output folder
    fig.savefig("figures/figure_3f_deletions.png",dpi=300, bbox_inches = "tight")

    # kde2 = gaussian_kde(highest_ins_freq_list)
    fig2, ax2 = plt.subplots(1, 1)
    # x = np.linspace(0, 30, 150)
    ax2.hist(highest_ins_freq_list, bins = 15, density=False, color = 'blue', alpha = 0.5)
    # ax2.plot(x, kde2(x), linewidth=2, color='k')
    # Adds labels to the figure
    ax2.set_xlabel('Most frequent insertion genotype (%)')
    ax2.set_ylabel('Number of gRNAs')

    # Save figure to output folder
    fig2.savefig("figures/figure_3f_insertions.png",dpi=300, bbox_inches = "tight")


# Find the observed frequencies for the test targets
def find_observed_freqs(test_targets):
    # Get dataset
    master_data = pickle.load(open('input/inDelphi_counts_and_deletion_features.pkl', 'rb'))
    counts = master_data['counts'].drop('fraction', axis=1)
    del_features = master_data['del_features']
    data = pd.concat((counts, del_features), axis=1).reset_index()

    # Get exp data
    data['Sample_Name'] = data['Sample_Name'].map(lambda x: str.split(x, "_")[-1])
    data = data[data["Sample_Name"].isin(list(test_targets.keys()))]
    exps = test_targets
    freqs_dict = {}

    for exp in exps:
      exp_data = data[data["Sample_Name"] == exp]
      exp_data.dropna(subset=["countEvents"], inplace=True)

      # Filter the dataframe to include only deletions (below 28 not sure if we need this though) and 1bp insertions
      exp_data = exp_data[((exp_data["Indel"].str.startswith("1+")) & (exp_data["Type"] == "INSERTION")) | ((exp_data["Type"] == "DELETION") & (exp_data["Size"] <= 28))]

      # Group the data by the "Indel" column and count the number of occurrences of each unique value
      counts = exp_data.groupby("Indel")["countEvents"].sum()

      # Calculate the total number of events in the dataframe
      total_events = exp_data["countEvents"].sum()

      # Divide the counts by the total number of rows in the dataframe to get the frequency as a percentage
      freqs = counts / total_events * 100
      exp_data["Frequencies (%)"] = exp_data["Indel"].map(freqs)
      exp_data.sort_values(by="Frequencies (%)", ascending=False, inplace=True)
      freq_per_indel = dict(zip(exp_data["Indel"], exp_data["Indel"].map(freqs)))

      freqs_dict[exps[exp]] = freq_per_indel
    return freqs_dict

if __name__ == '__main__':
    # Init inDelphi with mouse cell dataset
    inDelphi.init_model(celltype = 'mESC')

    # Get observed frequencies and test targets
    test_targets = {}
    with open('output_test/test_targets.csv') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            test_targets[row[0]] = row[1]
    observed_freqs = find_observed_freqs(test_targets)

    util.ensure_dir_exists("figures/")
    test_sequences = list(test_targets.values())[:4]
    generate_figure_1e(test_sequences, 27, observed_freqs)
    generate_figure_3f(test_targets)