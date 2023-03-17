# Author: maxwshen, https://github.com/maxwshen/indelphi-dataprocessinganalysis/blob/master/src-modeling-and-analysis/d2_model.py

# Model including a neural net in autograd
from __future__ import absolute_import, division
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.differential_operators import grad
from autograd.misc import flatten
from past.builtins import xrange 
import matplotlib
matplotlib.use('Pdf')
import matplotlib.pyplot as plt
from collections import defaultdict
import sys, string, pickle, subprocess, os, datetime
from mylib import util
import seaborn as sns, pandas as pd
from matplotlib.colors import Normalize
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, entropy
from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages 
import matplotlib.patches as mpatches
from Knn_helper import *

NAME = util.get_fn(__file__)

### Define neural network ###
def relu(x):       return np.maximum(0, x)
def sigmoid(x):    return 0.5 * (np.tanh(x) + 1.0)
def logsigmoid(x): return x - np.logaddexp(0, x)
def leaky_relu(x): return np.maximum(0, x) + np.minimum(0, x) * 0.001

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

# Do forward step (ie compute ouput from input) in the neural network.
def nn_match_score_function(params, inputs):
  # """Params is a list of (weights, bias) tuples.
  #    inputs is an (N x D) matrix."""
  inpW, inpb = params[0]
  # inputs = swish(np.dot(inputs, inpW) + inpb)
  inputs = sigmoid(np.dot(inputs, inpW) + inpb)
  # inputs = leaky_relu(np.dot(inputs, inpW) + inpb)
  for W, b in params[1:-1]:
    outputs = np.dot(inputs, W) + b
    # inputs = swish(outputs)
    inputs = sigmoid(outputs)
    # inputs = logsigmoid(outputs)
    # inputs = leaky_relu(outputs)
  outW, outb = params[-1]
  outputs = np.dot(inputs, outW) + outb
  return outputs.flatten()

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
    mh_scores = nn_match_score_function(nn_params, inp[idx])
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
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
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
    nn2_scores = nn_match_score_function(nn2_params, dls)
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

    # Calculate mh_total
    mh_phi_total = mh_phi_total._value if not isinstance(mh_phi_total, float) else mh_phi_total
    mh_less_phi_total = mh_less_phi_total._value if not isinstance(mh_less_phi_total, float) else mh_less_phi_total  
    mh_total = mh_phi_total + mh_less_phi_total
    
    normalized_del_freq_list = []
    for dl_freq in normalized_fq:
      if isinstance(dl_freq, float):
        normalized_del_freq_list.append(dl_freq)
      else:
        normalized_del_freq_list.append(dl_freq._value)
    # The "1 - " part is implemented in the inDelphi.py file.
    precision_score =  entropy(normalized_del_freq_list) / np.log(len(normalized_del_freq_list))
    
    # Append to list for storing
    knn_features.append([NAMES[idx], mh_total, precision_score])
  # Fix it.
  column_names = ["exp", "total_del_phi", "precision_score_dl"]
  knn_features_df = pd.DataFrame(knn_features, columns=column_names)
  knn_features_df.to_pickle(out_dir_params + 'knn_features_from_loss_function.pkl')
    # L2-Loss
    # LOSS += np.sum((normalized_fq - obs[idx])**2)
  return LOSS / num_samples

##
# Regularization 
##


# Backpropagation step.
##
# ADAM Optimizer
##
def exponential_decay(step_size):
  if step_size > 0.001:
      step_size *= 0.999
  return step_size

def adam_minmin(grad_both, init_params_nn, init_params_nn2, callback=None, num_iters=100, step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
  x_nn, unflatten_nn = flatten(init_params_nn)
  x_nn2, unflatten_nn2 = flatten(init_params_nn2)

  m_nn, v_nn = np.zeros(len(x_nn)), np.zeros(len(x_nn))
  m_nn2, v_nn2 = np.zeros(len(x_nn2)), np.zeros(len(x_nn2))
  for i in range(num_iters):
    g_nn_uf, g_nn2_uf = grad_both(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    g_nn, _ = flatten(g_nn_uf)
    g_nn2, _ = flatten(g_nn2_uf)

    if callback: 
      callback(unflatten_nn(x_nn), unflatten_nn2(x_nn2), i)
    
    step_size = exponential_decay(step_size)

    # Update parameters
    m_nn = (1 - b1) * g_nn      + b1 * m_nn  # First  moment estimate.
    v_nn = (1 - b2) * (g_nn**2) + b2 * v_nn  # Second moment estimate.
    mhat_nn = m_nn / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn = v_nn / (1 - b2**(i + 1))
    x_nn = x_nn - step_size * mhat_nn / (np.sqrt(vhat_nn) + eps)

    # Update parameters
    m_nn2 = (1 - b1) * g_nn2      + b1 * m_nn2  # First  moment estimate.
    v_nn2 = (1 - b2) * (g_nn2**2) + b2 * v_nn2  # Second moment estimate.
    mhat_nn2 = m_nn2 / (1 - b1**(i + 1))    # Bias correction.
    vhat_nn2 = v_nn2 / (1 - b2**(i + 1))
    x_nn2 = x_nn2 - step_size * mhat_nn2 / (np.sqrt(vhat_nn2) + eps)
  return unflatten_nn(x_nn), unflatten_nn2(x_nn2)


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

  # TODO : Fix this
def copy_script(out_dir):
  src_dir = '/cluster/mshen/prj/mmej_figures/src/'
  script_nm = __file__
  subprocess.call('cp ' + src_dir + script_nm + ' ' + out_dir, shell = True)
  return

def print_and_log(text, log_fn):
  with open(log_fn, 'a') as f:
    f.write(text + '\n')
  print(text)
  return

##
# Plotting and Writing
##
def save_parameters(nn_params, nn2_params, out_dir_params, letters):
  pickle.dump(nn_params, open(out_dir_params + letters + '_nn.pkl', 'wb'))
  pickle.dump(nn2_params, open(out_dir_params + letters + '_nn2.pkl', 'wb'))
  return

def rsq(nn_params, nn2_params, inp, obs, obs2, del_lens, num_samples, rs):
  rsqs1, rsqs2 = [], []
  for idx in range(len(inp)):
    ##
    # MH-based deletion frequencies
    ##
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    
    # Add MH-less contribution at full MH deletion lengths
    mh_vector = inp[idx].T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    for jdx in range(len(mh_vector)):
      if del_lens[idx][jdx] == mh_vector[jdx]:
        dl = del_lens[idx][jdx]
        mhless_score = nn_match_score_function(nn2_params, np.array(dl))
        mhless_score = np.exp(mhless_score - 0.25*dl)
        mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
        mhfull_contribution = mhfull_contribution + mask
    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))

    rsq1 = pearsonr(normalized_fq, obs[idx])[0]**2
    rsqs1.append(rsq1)

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = nn_match_score_function(nn2_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

    # iterate through del_lens vector, adding mh_scores (already computed above) to the correct index
    mh_contribution = np.zeros(28,)
    for jdx in range(len(Js)):
      dl = Js[jdx]
      if dl > 28:
        break
      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution
    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))


    rsq2 = pearsonr(normalized_fq, obs2[idx])[0]**2
    rsqs2.append(rsq2)

  return rsqs1, rsqs2

def save_rsq_params_csv(nms, rsqs, nn2_params, out_dir, iter_nm, data_type):
  with open(out_dir + iter_nm + '_' + data_type + '_rsqs_params.csv', 'w') as f:
    f.write( ','.join(['Exp', 'Rsq']) + '\n')
    for i in range(len(nms)):
      f.write( ','.join([nms[i], str(rsqs[i])]) + '\n' )
  return

def save_train_test_names(train_nms, test_nms, out_dir):
  with open(out_dir + 'train_exps.csv', 'w') as f:
    f.write( ','.join(['Exp']) + '\n')
    for i in range(len(train_nms)):
      f.write( ','.join([train_nms[i]]) + '\n' )
  with open(out_dir + 'test_exps.csv', 'w') as f:
    f.write( ','.join(['Exp']) + '\n')
    for i in range(len(test_nms)):
      f.write( ','.join([test_nms[i]]) + '\n' )
  return

def plot_mh_score_function(nn_params, out_dir, letters):
  data = defaultdict(list)
  col_names = ['MH Length', 'GC', 'MH Score']
  # Add normal MH
  for ns in range(5000):
    length = np.random.choice(range(1, 28+1))
    gc = np.random.uniform()
    features = np.array([length, gc])
    ms = nn_match_score_function(nn_params, features)[0]
    data['Length'].append(length)
    data['GC'].append(gc)
    data['MH Score'].append(ms)
  df = pd.DataFrame(data)

  with PdfPages(out_dir + letters + '_matchfunction.pdf', 'w') as pdf:
    # Plot length vs. match score
    sns.violinplot(x = 'Length', y = 'MH Score', data = df, scale = 'width')
    plt.title('Learned Match Function: MH Length vs. MH Score')
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot GC vs match score, color by length
    palette = sns.color_palette('hls', max(df['Length']) + 1)
    for length in range(1, max(df['Length'])+1):
      ax = sns.regplot(x = 'GC', y = 'MH Score', data = df.loc[df['Length']==length], color = palette[length-1], label = 'Length: %s' % (length))
    plt.legend(loc = 'best')
    plt.xlim([0, 1])
    plt.title('GC vs. MH Score, colored by MH Length')
    pdf.savefig()
    plt.close()
  return

def plot_pred_obs(nn_params, nn2_params, inp, obs, del_lens, nms, datatype, letters):
  num_samples = len(inp)
  [beta] = nn2_params
  pred = []
  obs_dls = []
  for idx in range(len(inp)):
    mh_scores = nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - beta*Js)
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))
    curr_pred = np.zeros(28 - 1 + 1)
    curr_obs = np.zeros(28 - 1 + 1)
    for jdx in range(len(del_lens[idx])):
      dl_idx = int(del_lens[idx][jdx]) - 1
      curr_pred[dl_idx] += normalized_fq[jdx]
      curr_obs[dl_idx] += obs[idx][jdx]
    pred.append(curr_pred.flatten())
    obs_dls.append(curr_obs.flatten())

  ctr = 0
  with PdfPages(out_dir + letters + '_' + datatype + '.pdf', 'w') as pdf:
    for idx in range(num_samples):
      ymax = max(max(pred[idx]), max(obs_dls[idx])) + 0.05
      rsq = pearsonr(obs_dls[idx], pred[idx])[0]**2

      plt.subplot(211)
      plt.title('Designed Oligo %s, Rsq=%s' % (nms[idx], rsq))
      plt.bar(range(1, 28+1), obs_dls[idx], align = 'center', color = '#D00000')
      plt.xlim([0, 28+1])
      plt.ylim([0, ymax])
      plt.ylabel('Observed')

      plt.subplot(212)
      plt.bar(range(1, 28+1), pred[idx], align = 'center', color = '#FFBA08')
      plt.xlim([0, 26+1])
      plt.ylim([0, ymax])
      plt.ylabel('Predicted')

      pdf.savefig()
      plt.close()
      ctr += 1
      if ctr >= 50:
        break
  return


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
    total_deletion_events = sum(exp_data['countEvents'])
    dl_freq_data = exp_data[exp_data['Size'] <= 28]
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
  num_folds = count_num_folders(out_place)
  out_letters = alphabetize(num_folds + 1)
  out_dir = out_place + out_letters + '/'
  out_dir_params = out_place + out_letters + '/parameters/'
  util.ensure_dir_exists(out_dir)
  copy_script(out_dir)
  util.ensure_dir_exists(out_dir_params)


  log_fn = out_dir + '_log_%s.out' % (out_letters)
  with open(log_fn, 'w') as f:
    pass
  print_and_log('out dir: ' + out_letters, log_fn)

  counter = 0
  seed = npr.RandomState(1)

  '''
  Model hyper-parameters
  '''
  nn_layer_sizes = [2, 16, 16, 1]
  nn2_layer_sizes = [1, 16, 16, 1]

  print_and_log("Loading data...", log_fn)
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
  save_train_test_names(NAMES_train, NAMES_test, out_dir)
   
  knn_features = pd.read_pickle('outputaab/parameters/knn_features_from_loss_function.pkl') 
  print(knn_features)
  
  # Train KNN
  print('starting KNN training')
  train_knn(knn_features, data)

  
  ''' 
  Training parameters
  '''
  param_scale = 0.1
  num_epochs = 30
  step_size = 0.10

  init_nn_params = init_random_params(param_scale, nn_layer_sizes, rs = seed)
  # init_nn_params = pickle.load(open('/cluster/mshen/prj/mmej_manda/out/2017-08-23/i2_model_mmh/aax/parameters/aav_nn.pkl'))

  init_nn2_params = init_random_params(param_scale, nn2_layer_sizes, rs = seed)

  # batch_size = len(INP_train)   # use all of training data
  batch_size = 200
  num_batches = int(np.ceil(len(INP_train) / batch_size))
  def batch_indices(iter):
    idx = iter % num_batches
    return slice(idx * batch_size, (idx+1) * batch_size)

  def objective(nn_params, nn2_params, iter):
    idx = batch_indices(iter)
    return main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)

  both_objective_grad = grad(objective, argnum=[0,1])

  def print_perf(nn_params, nn2_params, iter):
    print_and_log(str(iter), log_fn)
    if iter % 5 != 0:
      return None
    
    train_loss = main_objective(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    test_loss = main_objective(nn_params, nn2_params, INP_test, OBS_test, OBS2_train, DEL_LENS_test, len(INP_test), seed)

    tr1_rsq, tr2_rsq = rsq(nn_params, nn2_params, INP_train, OBS_train, OBS2_train, DEL_LENS_train, batch_size, seed)
    te1_rsq, te2_rsq = rsq(nn_params, nn2_params, INP_test, OBS_test, OBS2_test, DEL_LENS_test, len(INP_test), seed)
    
    out_line = ' %s  | %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t| %.3f\t|' % (iter, train_loss, np.mean(tr1_rsq), np.mean(tr2_rsq), test_loss, np.mean(te1_rsq), np.mean(te2_rsq))
    print_and_log(out_line, log_fn)

    if iter % 20 == 0:
      letters = alphabetize(int(iter/10))
      print_and_log(" Iter | Train Loss\t| Train Rsq1\t| Train Rsq2\t| Test Loss\t| Test Rsq1\t| Test Rsq2", log_fn)
      print_and_log('%s %s %s' % (datetime.datetime.now(), out_letters, letters), log_fn)
      save_parameters(nn_params, nn2_params, out_dir_params, letters)
      # save_rsq_params_csv(NAMES_test, test_rsqs, nn2_params, out_dir, letters, 'test')
      if iter >= 10:
      # if iter >= 0:
        pass
        # plot_mh_score_function(nn_params, out_dir, letters + '_nn')
        # plot_pred_obs(nn_params, nn2_params, INP_train, OBS_train, DEL_LENS_train, NAMES_train, 'train', letters)
        # plot_pred_obs(nn_params, nn2_params, INP_test, OBS_test, DEL_LENS_test, NAMES_test, 'test', letters)

    return None

  optimized_params = adam_minmin(both_objective_grad,
                                  init_nn_params, 
                                  init_nn2_params, 
                                  step_size = step_size, 
                                  num_iters = num_epochs,
                                  callback = print_perf)

  print('Done')
