import matplotlib.pyplot as plt
import numpy as np
import pickle
import shap
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import seaborn as sns
from inDelphi import __nn_function
from mylib import util


# Define a class that wraps the frequencies prediction functions
class FrequencyPredictor(BaseEstimator):
    def __init__(self, nn_one_find_loss):
        self.nn_one_find_loss = nn_one_find_loss
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.nn_one_find_loss(X)
    

def nn_one_predict_mh_phi_score(inputs):
    global nn_one_params
    outputs = []
    for input in inputs:
        mhl, gcf, dls = input[0], input[1], input[2]
        nn_input = np.array([mhl, gcf]).T  
        dls_input = np.array(dls).T
        mh_scores = __nn_function(nn_one_params, nn_input)
        Js = np.array(dls_input)
        unnormalized_fq = np.exp(mh_scores - 0.25 * Js)
        mh_phi_total = np.sum(unnormalized_fq, dtype=np.float64)
        outputs.append(mh_phi_total)

    return np.asarray(outputs)


def predict_del_genotype_freq_distribution(input):
    global nn_one_params
    global nn_two_params

    del_lens = np.split(input, [2], axis=1)[1].flatten()
    input = np.split(input, [2], axis=1)[0]

    mh_scores = __nn_function(nn_one_params, input)
    Js = np.array(del_lens)
    unnormalized_fq = np.exp(mh_scores - 0.25*Js)
    
    mh_vector = input.T[0]
    mhfull_contribution = np.zeros(mh_vector.shape)
    
    for jdx in range(len(mh_vector)):
      if del_lens[jdx] == mh_vector[jdx]:
          dl = del_lens[jdx]
          mhless_score = __nn_function(nn_two_params, np.array(dl))
          mhless_score = np.exp(mhless_score - 0.25*dl)
          mask = np.concatenate([np.zeros(jdx,), np.ones(1,) * mhless_score, np.zeros(len(mh_vector) - jdx - 1,)])
          mhfull_contribution = mhfull_contribution + mask

    unnormalized_fq = unnormalized_fq + mhfull_contribution
    normalized_fq = np.divide(unnormalized_fq, np.sum(unnormalized_fq))
    return normalized_fq


def predict_del_length_freq_distribution(input):
    global nn_one_params
    global nn_two_params

    del_lens = np.split(input, [2], axis=1)[1].flatten()
    input = np.split(input, [2], axis=1)[0]

    mh_scores = __nn_function(nn_one_params, input)
    Js = np.array(del_lens)

    dls = np.arange(1, 28+1)
    dls = dls.reshape(28, 1)
    nn2_scores = __nn_function(nn_two_params, dls)
    unnormalized_nn2 = np.exp(nn2_scores - 0.25*np.arange(1, 28+1))

    mh_contribution = np.zeros(28,)
    for jdx in range(len(Js)):
      dl = int(Js[jdx])
      if dl > 28:
        break

      mhs = np.exp(mh_scores[jdx] - 0.25*dl)
      mask = np.concatenate([np.zeros(dl - 1,), np.ones(1, ) * mhs, np.zeros(28 - (dl - 1) - 1,)])
      mh_contribution = mh_contribution + mask
    unnormalized_nn2 = unnormalized_nn2 + mh_contribution

    normalized_fq = np.divide(unnormalized_nn2, np.sum(unnormalized_nn2))

    return normalized_fq


def nn_two_predict_mh_less_phi_score(inputs):
    global nn_two_params
    outputs = []
    for input in inputs:
        mhless_score = __nn_function(nn_two_params, np.asarray(input[0]))
        unnormalized_nn2 = np.exp(mhless_score - 0.25*input[0])
        outputs.append(unnormalized_nn2)
    return np.asarray(outputs)


def find_del_genotype_freq_permutation_feature_importance():
    francesco_rq_ans = pickle.load(open('francesco_rq_ans.pkl', 'rb'))
    _, INP_test, _, OBS_test, _, _, _, _, _, DEL_LENS_test = francesco_rq_ans
 
    del_genotype_freq_predictor = FrequencyPredictor(predict_del_genotype_freq_distribution)

    # Add the deletion length as a feature
    for i in range(len(INP_test)):
        INP_test[i] = np.concatenate([INP_test[i], DEL_LENS_test[i].reshape(-1, 1)], axis=1)

    importances_means = []
    # Find the feature importance for each target site (here the features are permuted together)
    for i in range(len(INP_test)):
        result = permutation_importance(del_genotype_freq_predictor, INP_test[i], OBS_test[i], scoring="r2", n_repeats=5, random_state=0)
        importances_means.append(result['importances_mean'])
    
    # Conver the list of arrays to a 2D numpy array
    importances_means = np.array(importances_means)

    
    """_, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(importances_means[:, 0], importances_means[:, 1], c=importances_means[:, 2], cmap='viridis')
    ax.set_xlabel('MH length')
    ax.set_ylabel('GC content')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Del length permutation feature importance mean')
    plt.savefig("scatter.png")"""

    # Density plot
    _, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(importances_means[:, 0], label="MH length", ax=ax)
    sns.kdeplot(importances_means[:, 1], label="GC content", ax=ax)
    sns.kdeplot(importances_means[:, 2], label="DEL length", ax=ax)
    ax.set_xlabel('Permutation feature importance mean (averaged per each target site)')
    ax.set_ylabel('Estimated pdf')
    ax.legend()
    plt.savefig("del_genotype_freq_feature_importance_estimated_pdf.png")


def find_del_length_freq_permutation_feature_importance():

    francesco_rq_ans = pickle.load(open('francesco_rq_ans.pkl', 'rb'))
    _, INP_test, _, _, _, OBS2_test, _, _, _, DEL_LENS_test = francesco_rq_ans
    # One for each target site, the observed frequency of each deletion genotype, 15 % of the total dataset ie around 300 target sites
    del_length_freq_predictor = FrequencyPredictor(predict_del_length_freq_distribution)

    # Add the deletion length as a feature
    for i in range(len(INP_test)):
        INP_test[i] = np.concatenate([INP_test[i], DEL_LENS_test[i].reshape(-1, 1)], axis=1)

    importances_means = []
    # Find the feature importance for each target site (here the features are permuted together)
    for i in range(len(INP_test)):
        result = permutation_importance(del_length_freq_predictor, INP_test[i], OBS2_test[i], scoring="r2", n_repeats=5, random_state=0)
        importances_means.append(result['importances_mean'])
    
    # Conver the list of arrays to a 2D numpy array
    importances_means = np.array(importances_means)

    # Density plot
    _, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(importances_means[:, 0], label="MH length", ax=ax)
    sns.kdeplot(importances_means[:, 1], label="GC content", ax=ax)
    sns.kdeplot(importances_means[:, 2], label="DEL length", ax=ax)
    ax.set_xlabel('Permutation feature importance mean (averaged per each target site)')
    ax.set_ylabel('Estimated pdf')
    ax.legend()
    plt.savefig("del_length_freq_feature_importance_estimated_pdf.png")


def find_SHAP_values():
    francesco_rq_ans = pickle.load(open('francesco_rq_ans.pkl', 'rb'))
    INP_train, INP_test, _, _, _, _, _, _, DEL_LENS_train, DEL_LENS_test = francesco_rq_ans
    # Ungroup the rows by target site, consider all the rows as independent from their target site
    del_feature_train = np.concatenate(DEL_LENS_train).ravel()
    mh_features_train = np.concatenate(INP_train).ravel().reshape((len(del_feature_train), 2))
  
    # Concatenate mh_features (GC content and MH length) and del_feature
    nn_one_train_inputs = np.c_[mh_features_train, del_feature_train]

    filter_arr = []
    for input in nn_one_train_inputs:
        if input[2] <= 28:
            filter_arr.append(True)
        else:
            filter_arr.append(False)
    
    nn_one_train_inputs = nn_one_train_inputs[filter_arr]

    # Do the same for the test set
    del_feature_test = np.concatenate(DEL_LENS_test).ravel()
    mh_features_test = np.concatenate(INP_test).ravel().reshape((len(del_feature_test), 2))
    nn_one_test_inputs = np.c_[mh_features_test, del_feature_test]

    sampled_nn_one_train_inputs = shap.sample(nn_one_train_inputs, len(del_feature_train) // 8)
    
    # nn1 explainer
    # pass the network (with the trained parameters) to the explainer, and the background (ie train inputs, to compute the mean for the SHAP values)
    nn_one_explainer = shap.Explainer(nn_one_predict_mh_phi_score, sampled_nn_one_train_inputs)
    nn_one_shap_values = nn_one_explainer(nn_one_test_inputs)
    nn_one_shap_values.feature_names = ['MH length', 'GC content', 'Deletion length']

    
    pickle.dump(sampled_nn_one_train_inputs, open('SHAP_nn_one_train_inputs.pkl', 'wb'))
    pickle.dump(nn_one_test_inputs, open('SHAP_nn_one_test_inputs.pkl', 'wb'))
    pickle.dump(nn_one_shap_values, open('SHAP_nn_one_shap_values.pkl', 'wb'))


    nn_two_train_inputs = del_feature_train.reshape((len(del_feature_train), 1))
    nn_two_test_inputs = del_feature_test.reshape((len(del_feature_test), 1))
    
    sampled_nn_two_train_inputs = shap.sample(nn_two_train_inputs, len(del_feature_train) // 8)

    # nn2 explainer
    # pass the network (with the trained parameters) to the explainer, and the background (to compute the mean for the SHAP values)
    nn_two_explainer = shap.Explainer(nn_two_predict_mh_less_phi_score, sampled_nn_two_train_inputs)
    nn_two_shap_values = nn_two_explainer(nn_two_test_inputs)

    pickle.dump(sampled_nn_two_train_inputs, open('SHAP_nn_two_train_inputs.pkl', 'wb'))
    pickle.dump(nn_two_test_inputs, open('SHAP_nn_two_test_inputs.pkl', 'wb'))
    pickle.dump(nn_two_shap_values, open('SHAP_nn_two_shap_values.pkl', 'wb'))


def save_SHAP_figures():
    shap_values_one = pickle.load(open('SHAP_nn_one_shap_values.pkl', 'rb'))
    X_test = pickle.load(open('SHAP_nn_one_test_inputs.pkl', 'rb'))

    shap_values_one.feature_names = ['MH length', 'GC content', 'Deletion length']
    shap.plots.beeswarm(shap_values_one, show=False)
    print(shap_values_one[:, 0].values)
    plt.scatter(shap_values_one[:, 0].values, shap_values_one[:, 1].values, c=shap_values_one[:, 2].values, cmap='viridis', alpha=0.5)
    plt.xlabel('SHAP value of MH length')
    plt.ylabel('SHAP value of GC content')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.savefig('nn1_beeswarm_.png')
    shap.plots.scatter(shap_values_one[:, 'MH length'], color=shap_values_one, show=False)
    plt.tight_layout()
    plt.savefig('nn1_scatter_MH_length.png')
    shap.plots.scatter(shap_values_one[:, 'GC content'], color=shap_values_one, show=False)
    plt.tight_layout()
    plt.savefig('nn1_scatter_GC_content.png')
    shap.plots.scatter(shap_values_one[:, 'Deletion length'], color=shap_values_one, show=False)
    plt.tight_layout()
    plt.savefig('nn1_scatter_DEL_length.png')
    shap.plots.bar(shap_values_one, show=False)
    plt.tight_layout()
    plt.savefig('nn1_bar.png')
    shap.plots.heatmap(shap_values_one[:1000], show=False)
    plt.tight_layout()
    plt.savefig('nn1_heatmap.png')
    shap.summary_plot(shap_values_one[:1000], show=False)
    plt.tight_layout()
    plt.savefig('nn1_summary_plot.png')
    shap_values_two = pickle.load(open('SHAP_nn_two_shap_values.pkl', 'rb'))
    shap_values_two.feature_names = ['Deletion length']
    shap.plots.beeswarm(shap_values_two, show=False)
    plt.tight_layout()
    plt.savefig('nn2_beeswarm.png')
    shap.plots.scatter(shap_values_two, show=False)
    plt.tight_layout()
    plt.savefig('nn2_scatter.png')

def find_SHAP_values_knn():
  model = pickle.load(open('model-mlbio/rate_model_v2.pkl', 'rb'))
  X = pickle.load(open('model-mlbio/X_knn.pkl', 'rb'))
  print(X)
  print(model)
  # Tranform the entropy into a precision score
  X[:, 4] = 1 - X[:, 4]
  explainer = shap.Explainer(model.predict, X)
  shap_values = explainer(X)
  pickle.dump(shap_values, open('shap_values_knn.pkl', 'wb'))
  

def save_SHAP_figures_knn():
  shap_values = pickle.load(open('shap_values_knn.pkl', 'rb'))
  shap_values.feature_names = ['-4G freq', '-4T freq', '-3A freq', '-3G freq', 'Precision score', 'DelScore (Total Phi)']
  shap.plots.beeswarm(shap_values, show=False)
  plt.tight_layout()

  plt.savefig('knn_beeswarm.png')
  plt.show()
  shap.plots.scatter(shap_values, show=False)
  plt.tight_layout()
  plt.savefig('knn_scatter.png')
  plt.show()
  shap.plots.bar(shap_values, show=False)
  plt.tight_layout()
  plt.savefig('knn_bar.png')
  plt.show()
  shap.plots.heatmap(shap_values, show=False)
  plt.tight_layout()
  plt.savefig('knn_heatmap.png')
  plt.show()
  shap.summary_plot(shap_values, show=False)
  plt.tight_layout()
  plt.savefig('knn_summary_plot.png')
  plt.show()

"""def save_train_and_test_data():
    inp_dir = './input/'

    master_data = pickle.load(open(inp_dir + 'inDelphi_counts_and_deletion_features.pkl', 'rb'))
    counts = master_data['counts'].drop('fraction', axis=1)
    del_features = master_data['del_features']
    data = pd.concat((counts, del_features), axis=1)

    [exps, mh_lens, gc_fracs, del_lens, freqs, dl_freqs] = parse_input_data(data)

    INP = []
    for mhl, gcf in zip(mh_lens, gc_fracs):
        inp_point = np.array([mhl, gcf]).T
        INP.append(inp_point)
    INP = np.array(INP)
    OBS = np.array(freqs)
    OBS2 = np.array(dl_freqs)
    global NAMES
    NAMES = np.array([str(s) for s in exps])
    DEL_LENS = np.array(del_lens)

    ans = train_test_split(INP, OBS, OBS2, NAMES, DEL_LENS, test_size = 0.15, random_state = npr.RandomState(1))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_dir + 'francesco_rq_ans.pkl', 'wb') as f:
        pickle.dump(ans, f)"""

if __name__ == '__main__':
    global nn_one_params, nn_two_params
    nn_one_params = pickle.load(open('model-mlbio/aae_nn.pkl', 'rb'))
    nn_two_params = pickle.load(open('model-mlbio/aae_nn2.pkl', 'rb'))

    find_del_length_freq_permutation_feature_importance()
    find_SHAP_values()
    save_SHAP_figures()
    find_SHAP_values_knn()
    save_SHAP_figures_knn()