# Author: maxwshen, https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/master/src-modeling-and-analysis/d2_model.py

# Model including a neural net in autograd
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.differential_operators import grad
import matplotlib
matplotlib.use('Pdf')
import pickle, datetime
from mylib import util
import pandas as pd
from scipy.stats import entropy
from sklearn.model_selection import train_test_split
import utilities as ut
import forward_step as fw
import backprop as bp

NAME = util.get_fn(__file__)

### Define neural network ###
# Define the neural network architecture
# Initialize random paramaters for the neural networks
def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
  """Build a list of (weights, biases) tuples,
     one for each layer in the net."""
  return [(scale * rs.randn(m, n),   # weight matrix
           scale * rs.randn(n))      # bias vector
          for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


# Batch normalize each nn layer output to speed up training. (not used)
def batch_normalize(activations):
  mbmean = np.mean(activations, axis=0, keepdims=True)
  return (activations - mbmean) / (np.std(activations, axis=0, keepdims=True) + 1)


# Compute loss. 
def main_objective(nn_params, nn2_params, inp, obs, obs2, del_lens, num_samples, rs):
  LOSS = 0

  # iterate over all target site:  : [
  # [[CTTTCACTTTATAGATTTAT_mhls][CTTTCACTTTATAGATTTAT_gcfs]]]
  knn_features = []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    # inp[idx]:  [[CTTTCACTTTATAGATTTAT_mhls][CTTTCACTTTATAGATTTAT_gcfs]]
    # Compute all the psi scores. 
    mh_scores = fw.nn_match_score_function(nn_params, inp[idx])
    # take all the deletion lenghts corresponding to that target site CTTTCACTTTATAGATTTAT
    Js = np.array(del_lens[idx])
    # compute the phi scores from psi scores penalizing on the deletion length Js.
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    # Sum all the mh phi scores for that target site.
    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)
    
    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0] # [CTTTCACTTTATAGATTTAT_mhls] ie array containg all microhomology length for the current target site we are considering
    # Create a vector containing as many entries as the n of rows for the current target site.
    mhfull_contribution = np.zeros(mh_vector.shape)
    
    # Go over all the microhomology lengths for that target site ie CTTTCACTTTATAGATTTAT
    for jdx in range(len(mh_vector)):
      # this bit is explained at line 866 of the supplementary methods pdf.
      # if it is a full microhomology, namely, if the deletion length is equal to the homology length.
      if del_lens[idx][jdx] == mh_vector[jdx]:
        #dl = deletion length for that particular instance/row of the dataset for that target site
        dl = del_lens[idx][jdx]
        # Compute the mhless score for that deletion length using the 2nd neural network
        mhless_score = fw.nn_match_score_function(nn2_params, np.array(dl))
        # add penalization on deletion length again.
        mhless_score = np.exp(mhless_score - 0.25*dl)
        # mask is a vector of all zeros, except in the position where we are currently at, where
        # the entry is gonna be the score from the 2nd neural network
        #ie: for the 2nd row (which is a full microhomology: [0 1.2 0 0 0 0 0 0 0]
        mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
        # then we sum those two vectors: here mhfull_contribution is all zeros,
        # so the sum is gonna be equal to the mask again
        mhfull_contribution = mhfull_contribution + mask

    # Line 866 of the supplementary methods pdf.
    # here we are actually summing the mh scores with the mhless scores for the full microhomology rows
    # using the mhfull_contribution computed earlier using the mask.
    # unnormalized_fq is gonna be an array/list/vector with a score for every microhomology
    # with as many entries as the n of rows for that particular target site
    unnormalized_fq = unnormalized_fq + mhfull_contribution

    # Line 871 of the supplementary methods pdf.
    # Here we are getting the actual genotype deletion frequency distribution 
    # by normalizing all the scores in order for them to sum to 1.
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    # each target site can have multiple/diverse microhomology, each microhomology corresponds to
    # a particular deletion genotype so from the score of the microhomology we can get to the likelihood/frequency of its particular deletion genotype
    # hence from the microhomology scores we can get the frequency distribution for the deletion genotype 
    
    # Pearson correlation squared loss
    # Take the mean predicted frequency of all deletion genotype for that target site.   
    x_mean = np.mean(normalized_fq)
    # Take the mean observed frequency of all deletion genotype for that target site.   
    y_mean = np.mean(obs[idx])
    # Compute covariance between the two random variables (i.e. frequency observed and frequency predicted)
    pearson_numerator = np.sum((normalized_fq - x_mean)*(obs[idx] - y_mean))
    # compute standard deviation of the frequency predicted RV
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean)**2))
    # compute standard deviation of the frequency observed RV
    pearson_denom_y = np.sqrt(np.sum((obs[idx] - y_mean)**2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    # r squared: pearson correlation squared 
    rsq = (pearson_numerator/pearson_denom)**2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

    #
    # I want to make sure nn2 never outputs anything negative. 
    # Sanity check during training.
    #

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = fw.nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))
    # Sum all the mh-less phi scores for that target site.
    mh_less_phi_total = np.sum(unnormalized_nn2, dtype=np.float64)
    
    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    # Create an array/vector of 28 entries, each for each deletion length considered.
    mh_contribution = np.zeros(28,)
    # Js contains all the deletion lengths for that target site ie CTTTCACTTTATAGATTTAT,
    # it's gonna have as many rows as the n of entries related to CTTTCACTTTATAGATTTAT
    # in the del_features dataset.
    # so we go over all those deletion lengths 
    for jdx in range(len(Js)):
      # Take the deletion length for the current iteration
      dl = Js[jdx]
      if dl > 28:
        break
      # Line 877-878 of the supplementary methods pdf.
      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask
    # sum mh and mhless scores as specified in line 877 and 878 of of the supplementary methods pdf.
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    # Compute the frequency for each deletion length by normalizing in order for all the frequencies to sum up to 1.
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

    # Pearson correlation squared loss
    # Again compute the pearson correlation between the 2 random variables:
    # predicted deletion frequency for each deletion length
    # observed deletion frequency for each deletion length
    x_mean = np.mean(normalized_fq)
    y_mean = np.mean(obs2[idx])
    pearson_numerator = np.sum((normalized_fq - x_mean)*(obs2[idx] - y_mean))
    pearson_denom_x = np.sqrt(np.sum((normalized_fq - x_mean)**2))
    pearson_denom_y = np.sqrt(np.sum((obs2[idx] - y_mean)**2))
    pearson_denom = pearson_denom_x * pearson_denom_y
    # squared pearson correlation
    rsq = (pearson_numerator/pearson_denom)**2
    neg_rsq = rsq * -1
    LOSS += neg_rsq

  # L2-Loss
  # LOSS += np.sum((normalized_fq - obs[idx])**2)
  return LOSS / num_samples

# Save kNN features
# The function is basically a copy of main_objective but
# only with the parts necessary for the kNN feature computation
def save_knn_features(nn_params, nn2_params, inp, del_lens):
  knn_features = []

  for idx in range(len(inp)):
    mh_scores = fw.nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)
    
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = fw.nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25*dl)
        mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
        mhfull_contribution = mhfull_contribution + mask

    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = fw.nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))
    mh_less_phi_total = np.sum(unnormalized_nn2, dtype=np.float64)
    
    mh_contribution = np.zeros(28,)
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask

    # We need to use predictions from the second network
    # based on the authors code (file src-modeling-and-analysis/fi2_ins_ratio.py)
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

    #
    # Start calculating the kNN features
    # 

    # Calculate total phi
    mh_phi_total = mh_phi_total._value if not isinstance(mh_phi_total, float) else mh_phi_total
    mh_less_phi_total = mh_less_phi_total._value if not isinstance(mh_less_phi_total, float) else mh_less_phi_total  
    phi_total = mh_phi_total + mh_less_phi_total
    
    normalized_del_freq_list = []
    for dl_freq in normalized_fq:
      if isinstance(dl_freq, float):
        normalized_del_freq_list.append(dl_freq)
      else:
        normalized_del_freq_list.append(dl_freq._value)
        
    # The "1 - " part is implemented in the inDelphi.py file.
    precision_score =  entropy(normalized_del_freq_list) / np.log(len(normalized_del_freq_list))
    
    # Append to list for storing
    knn_features.append([NAMES[idx], phi_total, precision_score])

  column_names = ["exp", "total_del_phi", "precision_score_dl"]
  knn_features_df = pd.DataFrame(knn_features, columns=column_names)
  knn_features_df.to_pickle(out_dir_params + 'knn_features_from_loss_function.pkl')


def parse_input_data(data):
  # We care about deletions (MH and MH-less) for the neural networks.
  deletions_data = data[data['Type'] == 'DELETION'].reset_index()
  exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs = ([] for i in range(6))

  # To make this run in a short time, take only the first n elements (i.e. [:n])
  exps = deletions_data['Sample_Name'].unique()[:10]

  # Microhomology data has the homology length greater than 0
  mh_data = deletions_data[deletions_data['homologyLength'] != 0]

  for exp in exps:
    mh_exp_data = mh_data[mh_data['Sample_Name'] == exp]

    # These next 4 paramaters are related just to the mh deletions (see featurize function in c2_model_dataset.py)
    mh_lens.append(mh_exp_data['homologyLength'])
    gc_fracs.append(mh_exp_data['homologyGCContent'])
    del_lens.append(mh_exp_data['Size'].astype('int'))

    # Freqs determine how frequent microhomology deletions are for the given target site
    # We also need to normalize count events
    total_count_events = sum(mh_exp_data['countEvents'])
    # how frequent each deletion genotype (which is associated with a particular microhomology)
    # we get fromt the data is, normalized in order to sum to 1. 
    freqs.append(mh_exp_data['countEvents'].div(total_count_events))

    # compute how frequent each deletion length is for the given target site
    # both for microhomology and non microhomology deletion.
    exp_del_freqs = []
    exp_data = deletions_data[deletions_data['Sample_Name'] == exp]
    dl_freq_data = exp_data[exp_data['Size'] <= 28]
    total_deletion_events = sum(dl_freq_data['countEvents'])
    for del_len in range(1, 28+1):
      dl_freq = sum(dl_freq_data[dl_freq_data['Size'] == del_len]['countEvents']) / total_deletion_events
      exp_del_freqs.append(dl_freq)
    dl_freqs.append(exp_del_freqs)

  return [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs]
  

##
# Setup / Run Main
##
if __name__ == '__main__':
  out_place = './output'
  util.ensure_dir_exists(out_place)
  num_folds = ut.count_num_folders(out_place)
  out_letters = ut.alphabetize(num_folds + 1)
  out_dir = out_place + out_letters + '/'
  out_dir_params = out_place + out_letters + '/parameters/'
  util.ensure_dir_exists(out_dir)
  ut.copy_script(out_dir)
  util.ensure_dir_exists(out_dir_params)

  log_fn = out_dir + '_log_%s.out' % (out_letters)
  with open(log_fn, 'w') as f:
    pass
  ut.print_and_log('out dir: ' + out_letters, log_fn)

  counter = 0
  seed = npr.RandomState(1)

  '''
  Model hyper-parameters
  '''
  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]

  ut.print_and_log("Loading data...", log_fn)
  inp_dir = './input/'

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
  # mh_lens contains as many entries as exps, each entry is an ARRAY containing all the 
  # microhomology length in the dataset for that exp/target sequency, similarly 
  # gc_frac is an array of arrays, each array entry regards a specific target sequence and the 
  # relative GC conten observed. For the same target sequence we can observe different microhomologies
  # and different GC contents, see for example https://github.com/francescoopiccoli/MLBio_project/blob/main/input/del_features_example.csv  
  [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_input_data(data)
  INP = []
  for mhl, gcf in zip(mh_lens, gc_fracs):
    #mhl and gcf are both arrays
    inp_point = np.array([mhl, gcf]).T   # N * 2
    INP.append(inp_point)
  INP = np.array(INP)   # 2000 * N * 2
  # Neural network considers each N * 2 input, transforming it into N * 1 output.
  OBS = np.array(freqs)
  OBS2 = np.array(dl_freqs)
  global NAMES
  NAMES = np.array([str(s) for s in exps])
  DEL_LENS = np.array(del_lens)

  ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size = 0.15, random_state = seed)
  INP_train, INP_test, OBS_train, OBS_test, OBS2_train, OBS2_test, NAMES_train, NAMES_test, DEL_LENS_train, DEL_LENS_test = ans
  ut.save_train_test_names(NAMES_train, NAMES_test, out_dir)
  
  ''' 
  Training parameters
  '''
  param_scale = 0.1
  num_epochs = 30
  step_size = 0.10

  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs = seed)
  init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs = seed)

  batch_size = 200
  num_batches = int(np.ceil(len(INP_train) / batch_size))
  objective = lambda nn_params, nn2_params, iter: main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
  both_objective_grad = grad(objective, argnum=[0,1])

  def print_perf(nn_params, nn2_params, iter):
    ut.print_and_log(str(iter), log_fn)
    if iter % 5 != 0:
      return None
    
    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

    tr1_rsq, tr2_rsq = ut.rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    te1_rsq, te2_rsq = ut.rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)
    
    out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
    ut.print_and_log(out_line, log_fn)

    if iter % 20 == 0:
      letters = ut.alphabetize(int(iter/10))
      ut.print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2", log_fn)
      ut.print_and_log('%s %s %s' % (datetime.datetime.now(), out_letters, letters), log_fn)
      ut.save_parameters(nn_params, nn2_params, out_dir_params, letters)
      # save_rsq_params_csv(NAMES_test, test_rsqs, nn2_params, out_dir, letters, 'test')
      if iter >= 10:
      # if iter >= 0:
        pass
        # plot_mh_score_function(nn_params, out_dir, letters + '_nn')
        # plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train, 'train', letters)
        # plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test, 'test', letters)

    return None

  optimized_params = bp.adam_minmin(both_objective_grad,
                                  init_nn_params, 
                                  init_nn2_params, 
                                  step_size = step_size, 
                                  num_iters = num_epochs,
                                  callback = print_perf)

  print('NN_1 and NN_2 successfully trained!')

  print('Start kNN training')
  save_knn_features(optimized_params[0], optimized_params[1], INP, DEL_LENS)
  test = pd.read_pickle('outputaab/parameters/knn_features_from_loss_function.pkl')  
  print(test)
  print('kNN features successfully calculated!')
