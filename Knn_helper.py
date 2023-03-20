#Source: https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/635a10c4da2415173c24243230faedca32a7092c/src-modeling-and-analysis/e5_ins_ratebpmodel.py

from __future__ import division
#import _config, _lib, _data, _predict, _predict2
import sys, os, pickle
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
# Functions
##
def convert_oh_string_to_nparray(input):
    input = input.replace('[', '').replace(']', '')
    nums = input.split(' ')
    return np.array([int(s) for s in nums])

def featurize(rate_stats, Y_nm):
    fivebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Fivebase_OH']])
    threebases = np.array([convert_oh_string_to_nparray(s) for s in rate_stats['Threebase_OH']])

    total_del_phis = np.array(rate_stats['total_del_phi']).reshape(len(rate_stats['total_del_phi']), 1)
    precision_scores_dl = np.array(rate_stats['Del precision_scores_dl']).reshape(len(rate_stats['precision_scores_dl']), 1)
    print(total_del_phis.shape, fivebases.shape, precision_scores_dl.shape)

    Y = np.array(rate_stats[Y_nm])
    print(Y_nm)
    
    Normalizer = [(np.mean(fivebases.T[2]),
                      np.std(fivebases.T[2])),
                  (np.mean(fivebases.T[3]),
                      np.std(fivebases.T[3])),
                  (np.mean(threebases.T[0]),
                      np.std(threebases.T[0])),
                  (np.mean(threebases.T[2]),
                      np.std(threebases.T[2])),
                  (np.mean(total_del_phis),
                      np.std(total_del_phis)),
                  (np.mean(precision_scores_dl),
                      np.std(precision_scores_dl)),
                 ]

    fiveG = (fivebases.T[2] - np.mean(fivebases.T[2])) / np.std(fivebases.T[2])
    fiveT = (fivebases.T[3] - np.mean(fivebases.T[3])) / np.std(fivebases.T[3])
    threeA = (threebases.T[0] - np.mean(threebases.T[0])) / np.std(threebases.T[0])
    threeG = (threebases.T[2] - np.mean(threebases.T[2])) / np.std(threebases.T[2])
    gtag = np.array([fiveG, fiveT, threeA, threeG]).T

    total_del_phis = (total_del_phis - np.mean(total_del_phis)) / np.std(total_del_phis)
    precision_scores_dl = (precision_scores_dl - np.mean(precision_scores_dl)) / np.std(precision_scores_dl)

    X = np.concatenate(( gtag, total_del_phis, precision_scores_dl), axis = 1)
    feature_names = ['5G', '5T', '3A', '3G', 'Entropy', 'DelScore']
    print('Num. samples: %s, num. features: %s' % X.shape)

    return X, Y, Normalizer

##
# Train KNN model, X=training data, Y=target values, bp_stats
##

def generate_models(X, Y, bp_stats, Normalizer):

  # Train rate model
  model = KNeighborsRegressor()
  model.fit(X, Y)
  with open(out_dir + 'rate_model_v2.pkl', 'w') as f:
    pickle.dump(model, f)

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

  with open(out_dir + 'bp_model_v2.pkl', 'w') as f:
    pickle.dump(bp_model, f)

  with open(out_dir + 'Normalizer_v2.pkl', 'w') as f:
    pickle.dump(Normalizer, f)

  return


##
# Run statistics
##
def calc_statistics(df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions


  #Should be always 1
  editing_rate = 1
  alldf_dict['Editing Rate'].append(editing_rate)


  # # Denominator is ins
  # if sum(_lib.crispr_subset(df)['Count']) <= 1000:
  #   return

  # editing_rate = sum(_lib.crispr_subset(df)['Count']) / sum(_lib.notnoise_subset(df)['Count'])
  # alldf_dict['Editing Rate'].append(editing_rate)


  ins_criteria = (df['Type'] == 'INSERTION') # & (dcount_and_deletion_dff['Length'] == 1)
  ins_count = sum(df[ins_criteria]['countEvents'])

  del_criteria = (df['Type'] == 'DELETION')
  del_count = sum(df[del_criteria]['countEvents'])

  # TODO: Necessary? Why?
  # if del_count == 0:
    # return

  alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))

  # Store the Fivebase and threebase in df
  # Both normally and one-hot encoded

  print(len(exp))
  # TODO: This is hardcoded
  # Sample + sequence length is 28, (of which 20 for the sequence)
  cutsite = (int) (len(exp) - 10)

  # Get fifth base and encode it
  fivebase = exp[cutsite - 1]
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
  threebase = exp[cutsite]
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

  alldf_dict['_Experiment'].append(exp)

  return alldf_dict

def prepare_statistics(count_and_deletion_df):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  alldf_dict_1bp = defaultdict(list)

  timer = util.Timer(total = len(count_and_deletion_df))

  #TODO: Not sure if this is the idea?
  insertion_data = count_and_deletion_df[count_and_deletion_df['Type'] == 'INSERTION'].reset_index()

  # To make this run in a short time, take only the first n elements (i.e. [:n])
  exps = insertion_data['Sample_Name'].unique()[:10]

  for exp in exps:
    all_data = count_and_deletion_df[count_and_deletion_df['Sample_Name'] == exp]
    ins_data = insertion_data[insertion_data['Sample_Name'] == exp]
    print("seq: " + exp)
    calc_statistics(all_data, exp, alldf_dict)
    # calc_statistics_1bp(ins_data, exp, alldf_dict_1bp)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  return pd.DataFrame(alldf_dict), pd.DataFrame(alldf_dict_1bp)

##
# Run statistics
##
def calc_statistics_1bp(df, exp, alldf_dict):
  pass

##
# Train the KNN model
##
def train_knn(knn_features, count_and_deletion_df):
  print('Generating KNN Model')
  global out_dir
  util.ensure_dir_exists(out_dir)

 
  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()  

  #TODO: Calculcate bp_stats correctly, need results from NNs
  rate_stats, bp_stats = prepare_statistics(count_and_deletion_df)
  print(rate_stats.sample(n=5))
  print(bp_stats.sample(n=5))
  

  print(all_rate_stats)  
  all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
  all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

  #TODO: Get Entropy somehow?
  # rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
  
  # Check better the output of this merge operation
  all_rate_stats = pd.merge(all_rate_stats, knn_features, left_on='exp', right_on='exp', how='left')

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, all_bp_stats, Normalizer)

  return
