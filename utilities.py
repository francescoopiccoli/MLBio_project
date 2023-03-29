import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
from scipy.stats import pearsonr
from collections import defaultdict
from matplotlib.backends.backend_pdf import PdfPages 
import forward_step as fw
import csv
from mylib import util


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

def copy_script(out_dir):
  src_dir = ''
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
    mh_scores = fw.nn_match_score_function(nn_params, inp[idx])
    Js = np.array(del_lens[idx])
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    
    # Add MH-less contribution at full MH deletion lengths
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

    rsq1 = pearsonr(normalized_fq, obs[idx])[0]**2
    rsqs1.append(rsq1)

    ##
    # Deletion length frequencies, only up to 28
    #   (Restricts training to library data, else 27 bp.)
    ##
    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = fw.nn_match_score_function(nn2_params, dls)
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

def save_test_targets(test_exps):
  # Save targets for each exp
  names_and_targets = {}
  with open('grna-libA.txt') as guides, open('targets-libA.txt') as targets:
    valid_guides = guides.readlines()
    valid_targets = targets.readlines()

    for i, line in enumerate(valid_guides):
      names_and_targets[line[:-1]] = valid_targets[i][:-1]

  exps_targets = []
  for exp in test_exps:
    sp = str.split(exp, "_")
    grna = sp[-1]
    if grna in names_and_targets:
      exps_targets.append([sp[-1], names_and_targets[grna]])

  with open('output_test/test_targets.csv', 'w') as f:
    write = csv.writer(f)
    write.writerows(exps_targets)

def plot_mh_score_function(nn_params, out_dir, letters):
  data = defaultdict(list)
  col_names = ['MH Length', 'GC', 'MH Score']
  # Add normal MH
  for ns in range(5000):
    length = np.random.choice(range(1, 28+1))
    gc = np.random.uniform()
    features = np.array([length, gc])
    ms = fw.nn_match_score_function(nn_params, features)[0]
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
    mh_scores = fw.nn_match_score_function(nn_params, inp[idx])
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
  out_place = './output'
  util.ensure_dir_exists(out_place)
  num_folds = count_num_folders(out_place)
  out_letters = alphabetize(num_folds + 1)
  out_dir = out_place + out_letters + '/'
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