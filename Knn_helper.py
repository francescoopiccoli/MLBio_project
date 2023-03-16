#Source: https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/635a10c4da2415173c24243230faedca32a7092c/src-modeling-and-analysis/e5_ins_ratebpmodel.py

from __future__ import division
#import _config, _lib, _data, _predict, _predict2
import sys, os, datetime, subprocess, math, pickle, imp, fnmatch
import random
sys.path.append('/cluster/mshen/')
import numpy as np
from collections import defaultdict
from mylib import util
from mylib import compbio
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


# Default params
DEFAULT_INP_DIR = '/cluster/mshen/prj/mmej_manda2/out/2017-10-27/mb_grab_exons/'
NAME = util.get_fn(__file__)
out_dir = _config.OUT_PLACE + NAME + '/'

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

  # Denominator is crispr activity
  df = _lib.crispr_subset(df)
  if sum(df['Count']) <= 500:
    return
  df['Frequency'] = _lib.normalize_frequency(df)

  criteria = (df['Category'] == 'ins') & (df['Length'] == 1)
  if sum(df[criteria]['Count']) <= 100:
    return
  freq = sum(df[criteria]['Frequency'])
  alldf_dict['Frequency'].append(freq)

  s = df[criteria]

  try:
    a_frac = sum(s[s['Inserted Bases'] == 'A']['Frequency']) / freq
  except TypeError:
    a_frac = 0
  alldf_dict['A frac'].append(a_frac)

  try:
    c_frac = sum(s[s['Inserted Bases'] == 'C']['Frequency']) / freq
  except:
    c_frac = 0
  alldf_dict['C frac'].append(c_frac)

  try:
    g_frac = sum(s[s['Inserted Bases'] == 'G']['Frequency']) / freq
  except:
    g_frac = 0
  alldf_dict['G frac'].append(g_frac)

  try:
    t_frac = sum(s[s['Inserted Bases'] == 'T']['Frequency']) / freq
  except:
    t_frac = 0
  alldf_dict['T frac'].append(t_frac)

  seq, cutsite = _lib.get_sequence_cutsite(df)
  fivebase = seq[cutsite-1]
  alldf_dict['Base'].append(fivebase)

  alldf_dict['_Experiment'].append(exp)

  return alldf_dict

def prepare_statistics(data_nm):
  # Input: Dataset
  # Output: Uniformly processed dataset, requiring minimal processing for plotting but ideally enabling multiple plots
  # Calculate statistics associated with each experiment by name

  alldf_dict = defaultdict(list)

  dataset = _data.load_dataset(data_nm)
  if dataset is None:
    return

  timer = util.Timer(total = len(dataset))
  # for exp in dataset.keys()[:100]:
  for exp in dataset.keys():
    df = dataset[exp]
    calc_statistics(df, exp, alldf_dict)
    timer.update()

  # Return a dataframe where columns are positions and rows are experiment names, values are frequencies
  alldf = pd.DataFrame(alldf_dict)
  return alldf


##
# Load statistics from csv, or calculate 
##
def load_statistics(data_nm):
  print data_nm
  stats_csv_fn = out_dir + '%s.csv' % (data_nm)
  if not os.path.isfile(stats_csv_fn) or redo:
    print 'Running statistics from scratch...'
    stats_csv = prepare_statistics(data_nm)
    stats_csv.to_csv(stats_csv_fn)
  else:
    print 'Getting statistics from file...'
    stats_csv = pd.read_csv(stats_csv_fn, index_col = 0)
  print 'Done'
  return 

##
# Main
##
@util.time_dec
def main(data_nm = ''):
  print(NAME)
  global out_dir
  util.ensure_dir_exists(out_dir)

  import fi2_ins_ratio
  import fk_1bpins

  exps = ['VO-spacers-HEK293-48h-controladj', 
          'VO-spacers-K562-48h-controladj',
          'DisLib-mES-controladj',
          'DisLib-U2OS-controladj',
          'Lib1-mES-controladj'
         ]

  all_rate_stats = pd.DataFrame()
  all_bp_stats = pd.DataFrame()  
  for exp in exps:
    rate_stats = fi2_ins_ratio.load_statistics(exp)
    rate_stats = rate_stats[rate_stats['Entropy'] > 0.01]
    bp_stats = fk_1bpins.load_statistics(exp)
    exps = rate_stats['_Experiment']

    if 'DisLib' in exp:
      crit = (rate_stats['_Experiment'] >= 73) & (rate_stats['_Experiment'] <= 300)
      rs = rate_stats[crit]
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)

      crit = (rate_stats['_Experiment'] >= 16) & (rate_stats['_Experiment'] <= 72)
      rs = rate_stats[crit]
      rs = rs[rs['Ins1bp Ratio'] < 0.3] # remove outliers
      all_rate_stats = all_rate_stats.append(rs, ignore_index = True)

      crit = (bp_stats['_Experiment'] >= 73) & (bp_stats['_Experiment'] <= 300)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)

      crit = (bp_stats['_Experiment'] >= 16) & (bp_stats['_Experiment'] <= 72)
      rs = bp_stats[crit]
      all_bp_stats = all_bp_stats.append(rs, ignore_index = True)

    elif 'VO' in exp or 'Lib1' in exp:
      all_rate_stats = all_rate_stats.append(rate_stats, ignore_index = True)
      all_bp_stats = all_bp_stats.append(bp_stats, ignore_index = True)

  X, Y, Normalizer = featurize(all_rate_stats, 'Ins1bp/Del Ratio')
  generate_models(X, Y, all_bp_stats, Normalizer)

  return
