# Originally https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/635a10c4da2415173c24243230faedca32a7092c/src-modeling-and-analysis/e5_ins_ratebpmodel.py
# with some parts taken from https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/635a10c4da2415173c24243230faedca32a7092c/src-modeling-and-analysis/fi2_ins_ratio.py

from __future__ import division
import os, pickle
import numpy as np
from collections import defaultdict
from mylib import util
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor

##
# Setup environment
##
def alphabetize(num):
  assert num < 26**3, 'num bigger than 17576'
  mapper = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}
  hundreds = int(num / (26*26)) % 26
  tens = int(num / 26) % 26
  ones = num % 26
  return ''.join([mapper[hundreds], mapper[tens], mapper[ones]])

def count_num_folders(out_dir):
  for fold in os.listdir(out_dir):
    assert os.path.isdir(out_dir + fold), 'Not a folder!'
  return len(os.listdir(out_dir))

# Default params
out_place = './output'
util.ensure_dir_exists(out_place)
num_folds = count_num_folders(out_place)
out_letters = alphabetize(num_folds + 1)
out_dir = out_place + out_letters + '/'

##
# Helper functions
##
def convert_oh_string_to_nparray(input):
    return np.array([int(s) for s in input])

def featurize(rate_stats, Y_nm):
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
    threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)

    Y = np.array(rate_stats[Y_nm])
    
    Normalizer = [(np.mean(fivebases.T[2]),
                      np.std(fivebases.T[2])),
                  (np.mean(fivebases.T[3]),
                      np.std(fivebases.T[3])),
                  (np.mean(threebases.T[0]),
                      np.std(threebases.T[0])),
                  (np.mean(threebases.T[2]),
                      np.std(threebases.T[2])),
                  (np.mean(ent),
                      np.std(ent)),
                  (np.mean(del_scores),
                      np.std(del_scores)),
                 ]

    fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
    fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
    threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
    threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
    gtag = np.array([fiveG, fiveT, threeA, threeG]).T

    ent = (ent - np.mean(ent)) / np.std(ent)
    del_scores = (del_scores - np.mean(del_scores)) / np.std(del_scores)

    X = np.concatenate(( gtag, ent, del_scores), axis = 1)

    return X, Y, Normalizer

##
# Train KNN model, X=training data, Y=target values, bp_stats
##
def generate_models(X, Y, bp_stats, Normalizer):
  # Train rate model
  model = KNeighborsRegressor()
  model.fit(X, Y)
  with open(out_dir + 'rate_model_v2.pkl', 'wb') as f:
    pickle.dump(model, f)

  with open(out_dir + 'X_knn.pkl', 'wb') as f:
    pickle.dump(X, f)

  with open(out_dir + 'Y_knn.pkl', 'wb') as f:
    pickle.dump(Y, f)

  # Obtain bp stats
  bp_model = dict()
  ins_bases = ['A frac', 'C frac', 'G frac', 'T frac']
  t_melt = pd.melt(bp_stats, 
                   id_vars = ['Base'], 
                   value_vars = ins_bases, 
                   var_name = 'Ins Base', 
                   value_name = 'Fraction')
  for base in list('ACGT'):
    bp_model[base] = dict()
    mean_vals = []
    for ins_base in ins_bases:
      crit = (t_melt['Base'] == base) & (t_melt['Ins Base'] == ins_base)
      mean_vals.append(float(np.mean(t_melt[crit])))
    for bp, freq in zip(list('ACGT'), mean_vals):
      bp_model[base][bp] = freq / sum(mean_vals)

  with open(out_dir + 'bp_model_v2.pkl', 'wb') as f:
    pickle.dump(bp_model, f)

  with open(out_dir + 'Normalizer_v2.pkl', 'wb') as f:
    pickle.dump(Normalizer, f)

  return

##
# Prepare statistics
##
def prepare_statistics(knn_features, count_and_deletion_df):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)
  alldf_dict_1bp = defaultdict(list)

  exps = count_and_deletion_df['Sample_Name'].unique()
  timer = util.Timer(total = len(exps))
  
  for exp in exps:
    exp_data = count_and_deletion_df[count_and_deletion_df['Sample_Name'] == exp]
  
    calc_statistics(exp_data, exp, alldf_dict, count_and_deletion_df, knn_features)
    calc_statistics_1bp(exp_data, exp, alldf_dict_1bp)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  return pd.DataFrame(alldf_dict), pd.DataFrame(alldf_dict_1bp)

##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict, count_and_deletion_df, knn_features):
  if sum(df['countEvents']) <= 1000: return

  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions
  alldf_dict['_Experiment'].append(exp)

  # Equals to 1 due to absence of noise in our dataset
  editing_rate = 1
  alldf_dict['Editing Rate'].append(editing_rate)

  ins_criteria = (df['Type'] == 'INSERTION') & (df['Indel'].str.startswith('1+'))
  ins_count = sum(df[ins_criteria]['countEvents'])
  del_criteria = (df['Type'] == 'DELETION')
  del_count = sum(df[del_criteria]['countEvents'])
  mhdel_count = sum(df[(df['Type'] == 'DELETION') & (df['homologyLength'] != 0)]['countEvents'])
  alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))
  alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
  alldf_dict['Ins1bp Ratio'].append(ins_count / sum(df['countEvents']))

  # Get fifth base and encode it
  fivebase = exp[len(exp) - 4]
  alldf_dict['Fivebase'].append(fivebase)

  if fivebase == 'A':
    fivebase_oh = np.array([1, 0, 0, 0])
  if fivebase == 'C':
    fivebase_oh = np.array([0, 1, 0, 0])
  if fivebase == 'G':
    fivebase_oh = np.array([0, 0, 1, 0])
  if fivebase == 'T':
    fivebase_oh = np.array([0, 0, 0, 1])
  alldf_dict['Fivebase_OH'].append(fivebase_oh)

  # Get third base and encode it
  threebase = exp[len(exp) - 3]
  alldf_dict['Threebase'].append(threebase)
  if threebase == 'A':
    threebase_oh = np.array([1, 0, 0, 0])
  if threebase == 'C':
    threebase_oh = np.array([0, 1, 0, 0])
  if threebase == 'G':
    threebase_oh = np.array([0, 0, 1, 0])
  if threebase == 'T':
    threebase_oh = np.array([0, 0, 0, 1])
  alldf_dict['Threebase_OH'].append(threebase_oh)

  # Get deletion score and norm entropy
  del_score = 0
  norm_entropy = 0
  if len(knn_features[knn_features['exp'] == exp]) > 0:
    del_score = knn_features[knn_features['exp'] == exp]['total_del_phi'].values[0]
    norm_entropy = knn_features[knn_features['exp'] == exp]['precision_score_dl'].values[0]
  alldf_dict['Del Score'].append(del_score)
  alldf_dict['Entropy'].append(norm_entropy)

  return alldf_dict

##
# Run statistics
##
def calc_statistics_1bp(df, exp, alldf_dict):

  df["Frequency"] = df["countEvents"] / sum(df["countEvents"])
  one_bp_insertion_df = df[(df['Type'] == 'INSERTION') & (df['Indel'].str.startswith('1+'))]
  if sum(one_bp_insertion_df['countEvents']) <= 100:
    return
  
  alldf_dict['_Experiment'].append(exp)
  onebp_freq = sum(one_bp_insertion_df['Frequency'])
  try:
    a_frac = sum(one_bp_insertion_df[one_bp_insertion_df['Indel'].str.endswith('A')]['Frequency']) / onebp_freq
  except:
    a_frac = 0
  alldf_dict['A frac'].append(a_frac)

  try:
    c_frac = sum(one_bp_insertion_df[one_bp_insertion_df['Indel'].str.endswith('C')]['Frequency']) / onebp_freq
  except:
    c_frac = 0
  alldf_dict['C frac'].append(c_frac)

  try:
    g_frac = sum(one_bp_insertion_df[one_bp_insertion_df['Indel'].str.endswith('G')]['Frequency']) / onebp_freq
  except:
    g_frac = 0
  alldf_dict['G frac'].append(g_frac)

  try:
    t_frac = sum(one_bp_insertion_df[one_bp_insertion_df['Indel'].str.endswith('T')]['Frequency']) / onebp_freq
  except:
    t_frac = 0
  alldf_dict['T frac'].append(t_frac)

  # Get fifth base and encode it
  fivebase = exp[len(exp) - 4]
  alldf_dict['Base'].append(fivebase)
  return alldf_dict

def train_knn(count_and_deletion_df, knn_features):
  global out_dir
  util.ensure_dir_exists(out_dir)
  rate_stats, bp_stats = prepare_statistics(knn_features, count_and_deletion_df)
  rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]

  X, Y, Normalizer = featurize(rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, bp_stats, Normalizer)
