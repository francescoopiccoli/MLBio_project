#Source: https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/635a10c4da2415173c24243230faedca32a7092c/src-modeling-and-analysis/e5_ins_ratebpmodel.py

from __future__ import division
#import _config, _lib, _data, _predict, _predict2
import sys, os
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

    ent = np.array(rate_stats['Entropy']).reshape(len(rate_stats['Entropy']), 1)
    del_scores = np.array(rate_stats['Del Score']).reshape(len(rate_stats['Del Score']), 1)
    print(ent.shape, fivebases.shape, del_scores.shape)

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
    X = np.concatenate(( gtag, ent, del_scores), axis = 1)
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

  # ins_criteria = (df['Category'] == 'ins') & (df['Length'] == 1) & (df['Indel with Mismatches'] != 'yes')
  # ins_count = sum(df[ins_criteria]['Count'])

  # del_criteria = (df['Category'] == 'del') & (df['Indel with Mismatches'] != 'yes')
  # del_count = sum(df[del_criteria]['Count'])
  # if del_count == 0:
  #   return
  # alldf_dict['Ins1bp/Del Ratio'].append(ins_count / (del_count + ins_count))

  # mhdel_crit = (df['Category'] == 'del') & (df['Indel with Mismatches'] != 'yes') & (df['Microhomology-Based'] == 'yes')
  # mhdel_count = sum(df[mhdel_crit]['Count'])
  # try:
  #   alldf_dict['Ins1bp/MHDel Ratio'].append(ins_count / (mhdel_count + ins_count))
  # except ZeroDivisionError:
  #   alldf_dict['Ins1bp/MHDel Ratio'].append(0)

  # ins_ratio = ins_count / sum(_lib.crispr_subset(df)['Count'])
  # alldf_dict['Ins1bp Ratio'].append(ins_ratio)

  # seq, cutsite = _lib.get_sequence_cutsite(df)
  # fivebase = seq[cutsite - 1]
  # alldf_dict['Fivebase'].append(fivebase)

  # _predict2.init_model()
  # del_score = _predict2.total_deletion_score(seq, cutsite)
  # alldf_dict['Del Score'].append(del_score)

  # dlpred = _predict2.deletion_length_distribution(seq, cutsite)
  # from scipy.stats import entropy
  # norm_entropy = entropy(dlpred) / np.log(len(dlpred))
  # alldf_dict['Entropy'].append(norm_entropy)

  # local_seq = seq[cutsite - 4 : cutsite + 4]
  # gc = (local_seq.count('C') + local_seq.count('G')) / len(local_seq)
  # alldf_dict['GC'].append(gc)

  # Store the Fivebase and threebase in df
  # Both normally and one-hot encoded

  # Get cutside by diving sequence length by 2
  # TODO: Is this correct
  cutsite = (int) (len(exp) / 2)

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

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  alldf_dict_1bp = defaultdict(list)

  timer = util.Timer(total = len(data_nm))

  #TODO: Not sure if this is the idea?
  insertion_data = data_nm[data_nm['Type'] == 'INSERTION'].reset_index()

  # To make this run in a short time, take only the first n elements (i.e. [:n])
  exps = insertion_data['Sample_Name'].unique()[:10]

  for exp in exps:
    ins_data = insertion_data[insertion_data['Sample_Name'] == exp]
    print("seq: " + exp)
    calc_statistics(ins_data, exp, alldf_dict)
    calc_statistics_1bp(ins_data, exp, alldf_dict_1bp)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  return pd.DataFrame(alldf_dict), pd.DataFrame(alldf_dict_1bp)

##
# Run statistics
##
def calc_statistics_1bp(df, exp, alldf_dict):
  # Calculate statistics on df, saving to alldf_dict
  # Deletion positions


  # if sum(df['Count']) <= 500:
  #   return
  
  # df['Frequency'] = _lib.normalize_frequency(df)

  # criteria = (df['Category'] == 'ins') & (df['Length'] == 1)
  # if sum(df[criteria]['Count']) <= 100:
  #   return
  # freq = sum(df[criteria]['Frequency'])
  # alldf_dict['Frequency'].append(freq)

  # s = df[criteria]

  # freq = 1

  # try:
  #   a_frac = sum(s[s['Inserted Bases'] == 'A']['Frequency']) / freq
  # except TypeError:
  #   a_frac = 0
  # alldf_dict['A frac'].append(a_frac)

  # try:
  #   c_frac = sum(s[s['Inserted Bases'] == 'C']['Frequency']) / freq
  # except:
  #   c_frac = 0
  # alldf_dict['C frac'].append(c_frac)

  # try:
  #   g_frac = sum(s[s['Inserted Bases'] == 'G']['Frequency']) / freq
  # except:
  #   g_frac = 0
  # alldf_dict['G frac'].append(g_frac)

  # try:
  #   t_frac = sum(s[s['Inserted Bases'] == 'T']['Frequency']) / freq
  # except:
  #   t_frac = 0
  # alldf_dict['T frac'].append(t_frac)

  # cutsite = (int) (len(exp) / 2)

  # fivebase = seq[cutsite-1]
  # alldf_dict['Base'].append(fivebase)

  # alldf_dict['_Experiment'].append(exp)

  return alldf_dict


##
# Train the KNN model
##
def train_knn(knn_features, data_nm):
  print('Generating KNN Model')
  global out_dir
  util.ensure_dir_exists(out_dir)

  #import fk_1bpins

  
  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()  

  rate_stats, bp_stats = prepare_statistics(data_nm)
  print(rate_stats.sample(n=5))
  print(bp_stats.sample(n=5))
  
  #TODO: Get Entropy somehow?
  # rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]

  all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
  all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, all_bp_stats, Normalizer)

  return
