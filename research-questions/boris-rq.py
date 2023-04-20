from __future__ import absolute_import, division
from __future__ import print_function

import os
import sys
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.differential_operators import grad
import pickle, datetime
from mylib import util
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from training_model import init_random_params, main_objective, parse_input_data
import utilities as ut
from nn_helper import *
import matplotlib.pyplot as plt
from scipy import stats

NAME = util.get_fn(__file__)

if __name__ == '__main__':
  out_place = './output-boris-rq'
  util.ensure_dir_exists(out_place)

  print("Loading data...")
  inp_dir = '../input/'

  # A pickle file containing a dict object with two keys: counts and del_features.
  # The guide names are in the format <guide_id>_<guide sequence>. The guide sequence can be extracted from this id and used to determine the -3, -4 and -5 nucleotides for the kNN insertion model and insertion-type repair outcomes.
  master_data = pickle.load(open(inp_dir + 'inDelphi_counts_and_deletion_features.pkl', 'rb'))

  # counts: Contains a dataframe detailing the observed counts for each repair outcome (insertions and deletions) for every target sequence. The “fraction” column can be ignored.
  counts = master_data['counts'].drop('fraction', axis=1)

  # del_features: contains a dataframe detailing the deletion length, homology length, and homology GC content, for each deletion-type repair outcome for every target sequence.
  del_features = master_data['del_features']

  # merged counts and del_features
  data = pd.concat((counts, del_features), axis=1)

  # Unpack data from e11_dataset
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_input_data(data)
  
  INP = []
  for mhl, gcf in zip(mh_lens, gc_fracs):
    inp_point = np.array([mhl, gcf]).T   # N * 2
    INP.append(inp_point)

  INP = np.array(INP)   # 2000 * N * 2
  OBS = np.array(freqs)
  OBS2 = np.array(dl_freqs)
  global NAMES
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  ''' 
  Training parameters
  '''
  param_scale = 0.1
  num_epochs = 50 + 1
  step_size = 0.10

  ''' 
  Model hyperparameters of interest
  '''
  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]

  nn_layer_sizes_new = [2, 32, 32, 16, 1]
  nn2_layer_sizes_new = [1, 16, 16, 1]
  
  def ten_fold_crossvalidation(model_unchanged = True):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    model_folder = "unchanged-model" if model_unchanged else "modified-model"
    out_dir = f"{out_place}/ten-fold-crossvalidation/{model_folder}"
    util.ensure_dir_exists(out_dir)

    log_dest = f"{out_dir}/log.out"
    scores = []

    for i, (train_index, test_index) in enumerate(kf.split(INP)):
      seed = npr.RandomState(i + 1)
      out_fold_dir = f"{out_dir}/fold-{i + 1}/"
      util.ensure_dir_exists(out_fold_dir)

      INP_train, INP_test = INP[train_index], INP[test_index]
      OBS_train, OBS_test = OBS[train_index], OBS[test_index]
      OBS2_train, OBS2_test = OBS2[train_index], OBS2[test_index]
      NAMES_train, NAMES_test = NAMES[train_index], NAMES[test_index]
      DEL_LENS_train, DEL_LENS_test = DEL_LENS[train_index], DEL_LENS[test_index]

      nn_layer_sizes_exp = nn_layer_sizes if model_unchanged else nn_layer_sizes_new
      nn2_layer_sizes_exp = nn2_layer_sizes if model_unchanged else nn2_layer_sizes_new

      init_nn_params = init_random_params(param_scale, nn_layer_sizes_exp, rs = seed)
      init_nn2_params = init_random_params(param_scale, nn2_layer_sizes_exp, rs = seed)

      batch_size = 200
      num_batches = int(np.ceil(len(INP_train) / batch_size))
      objective = lambda nn_params, nn2_params, iter: main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
      both_objective_grad = grad(objective, argnum=[0,1])

      def print_perf(nn_params, nn2_params, iter):
          log_out_path = f"{out_fold_dir}_log.out"
          ut.print_and_log(str(iter), log_out_path)
          if iter % 5 != 0: return None
          
          train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
          test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

          tr1_rsq, tr2_rsq = ut.rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
          te1_rsq, te2_rsq = ut.rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)
          
          out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
          ut.print_and_log(out_line, log_out_path)

          if iter % 50 == 0 and iter != 0:
              ut.print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2", log_out_path)
              ut.print_and_log('%s' % (datetime.datetime.now()), log_out_path)
              ut.save_parameters(nn_params, nn2_params, out_fold_dir, f"iter_{iter}")
              ut.print_and_log(f"Fold {i + 1}: {test_loss}", log_dest)
              scores.append(test_loss)

      nn_params, nn2_params = adam_minmin(both_objective_grad, init_nn_params, init_nn2_params, step_size = step_size, num_iters = num_epochs, callback = print_perf)
      main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

    return scores
  

  # Perform crossvalidation
  print(f"------ Starting 10-fold cross-validation on both models ------")
  folds_unchanged = ten_fold_crossvalidation()
  folds_changed = ten_fold_crossvalidation(False)
  print(folds_unchanged)
  print(folds_changed)
  print(f"------ Cross-validation successfully finished! ------")

  diff = [x1 - folds_changed[i] for i, x1 in enumerate(folds_unchanged)]
  print(np.median(diff), np.mean(diff), np.std(diff), stats.skew(diff))

  # Output histogram and PDF of the fold differences
  n, bins, patches = plt.hist(diff, bins=10, density=True, alpha=0.5)
  x = np.linspace(min(bins), max(bins), 100)
  pdf = stats.norm.pdf(x, loc=np.mean(diff), scale=np.std(diff))

  plt.plot(x, pdf, color='red', linewidth=2)
  plt.xlabel('Data')
  plt.ylabel('Density')
  plt.show()

  # Output boxplot for the fold differences
  fig, ax = plt.subplots()
  ax.boxplot(diff)
  ax.set_ylabel('Values')
  plt.show()

  # Do sign test
  pos = sum(1 for d in diff if d > 0)
  neg = sum(1 for d in diff if d < 0)
  p_value = stats.binomtest(min(pos, neg), n=len(diff), alternative='two-sided')
  print("p-value:", p_value)

  # Do Wilcoxon test
  res = stats.wilcoxon(diff)
  print(f"Wilcoxon results (statistic, p-value): {res}")