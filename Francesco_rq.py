import matplotlib.pyplot as plt
import numpy as np
import pickle

import shap
import lime
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import seaborn as sns
from inDelphi import __nn_function

out_folder = "outputaab"


# Define a class that wraps nn_one_predict function
class NnOnePredictor(BaseEstimator):
    def __init__(self, nn_one_find_loss):
        self.nn_one_find_loss = nn_one_find_loss
    
    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.nn_one_find_loss(X)
    
# New idea: use the total phi score (sum of the phi score for MH and MHless) as prediction value of the two neural networks.
# In this way we can see how each feature (MH length and GC content and DEL length) contributes to the total phi score, namely to the 
# strength of the microhomology.
# It'd be a way to intepret the two nns together as the total phi score is the sum of the two phi scores.
# See if as expected the MH length and GC content contribute more to the total phi score than the DEL length.


# New idea: evaluate the whole model: use "prediction frequency" column of __predict_ins method as the prediction, the target.
# Pass as input the MH length, GC content and DEL length, all the other attributes for the knn we can figure them out from the inside.
# The SHAP or permutation feature importance will tell us how much of the 3 features contribute to the prediction frequency for each repair outcome.
# Is this feasible? Does this make sense?


def nn_one_predict_mh_score(inputs):
    global nn_one_params
    outputs = []
    # For each target site in the test set, consider them one at a time, and compute the mh_phi_total for each of them.
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

def nn_one_predict_freq_distribution(input):
    global nn_one_params
    global nn_two_params
    global obs
    global idx

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
    # Frequency distribution for each deletion genotype for that target site
    print(normalized_fq.shape)
    return normalized_fq

# Eventually what I am computing here is a prediction, namely the prediction for each deletion genotype
# so I am not actually computing the loss, the loss is computed by the permutation feature imortance function
# by passing the observed frequencies and the predicted frequencies to the r2_score function.

    
def nn_two_predict_mh_less_score(inputs):
    global nn_two_params
    outputs = []
    for input in inputs:
        mhless_score = __nn_function(nn_two_params, np.asarray(input[0]))
        outputs.append(mhless_score)
    return np.asarray(outputs)


# freq distribution on deletion genotype -> here we find out which features are most important for the prediction of the frequency distribution
# If I am interested in making a prediction on the frequency of each repair outcome (which is needed to compute the LOSS), I need the two frequencies distribution and
# I need to consider the target site, and process the rows of each target site together. So this is the case when I need to compute the loss
# fuction. 
# this is the case for permutation feature importance, where I need to compute the loss function to make it work.
# The permutation feature importance needs to compute the loss, so the mh score is not enough (differently from SHAP), indeed we need to find 
# the frequency distributions to compute the loss (which is the r2 score).
# So here we need to consider rows grouped by target site (a frequency is not related to one single row, but to all rows of the same target site)
def find_permutation_feature_importance():
    global obs, del_lens, idx 
    
    francesco_rq_ans = pickle.load(open('outputaab/francesco_rq_ans.pkl', 'rb'))
    _, INP_test, OBS_test, _, DEL_LENS_test = francesco_rq_ans
    # One for each target site, the observed frequency of each deletion genotype, 15 % of the total dataset ie around 300 target sites
    obs = OBS_test
    del_lens = DEL_LENS_test
    nn_one_predictor = NnOnePredictor(nn_one_predict_freq_distribution)
    # Find the mean of the permutation feature importance for each target site, for each of the two features

    # Add the deletion length as a feature
    for i in range(len(INP_test)):
        INP_test[i] = np.concatenate([INP_test[i], DEL_LENS_test[i].reshape(-1, 1)], axis=1)

    importances_means = []
    # Find the feature importance for each target site (here the features are permuted together)
    # When we permute two features together, we are essentially evaluating the importance of both features combined.
    for i in range(len(INP_test)):
        idx = i
        result = permutation_importance(nn_one_predictor, INP_test[i], OBS_test[i], scoring="r2", n_repeats=5, random_state=0)
        importances_means.append(result['importances_mean'])
    
    # Conver the list of arrays to a 2D numpy array
    importances_means = np.array(importances_means)

    # From the images it seemse like the importance of the 3 features is positvely correlated
    # GC content importance has a smaller variance, and its importance is lower than the other two features
    # MH length and DEL length have a similar importance, and a similar variance
    # Scatter plot
    fig, ax = plt.subplots(figsize=(8,6))
    scatter = ax.scatter(importances_means[:, 0], importances_means[:, 1], c=importances_means[:, 2], cmap='viridis')
    ax.set_xlabel('MH length')
    ax.set_ylabel('GC content')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Del length permutation feature importance mean')
    plt.savefig("scatter.png")

    # Density plot
    fig, ax = plt.subplots(figsize=(8,6))
    sns.kdeplot(importances_means[:, 0], label="MH length", ax=ax)
    sns.kdeplot(importances_means[:, 1], label="GC content", ax=ax)
    sns.kdeplot(importances_means[:, 2], label="DEL length", ax=ax)
    ax.set_xlabel('Permutation feature importance mean (averaged per each target site)')
    ax.set_ylabel('Estimated pdf')
    ax.legend()
    plt.savefig("density.png")


# high MH score -> strong microhomolgy -> here we understand which features contribute in making the MH score high
# If I want to understand how the mh score of the 1st neural network is influenced by the input parameters, then I dont need to consider
# the target site, I can consider the rows one at a time, and compute the mh score for each of them.
# This is the case for SHAP, when I am not intersted in the loss function, indeed I need only to compute the mh score prediction to find the SHAP
# values for each feature.
# We dont group by target site, we consider each single row irrespective of the target site
# The first neural network score (which I am using for the SHAP values) is computed for each row based on the GC content and MH length, and the deletion length
# Grouping the rows by target site, is useful when we are interested in the frequencies distribution, I am not using the frequencies distribution for the SHAP values
# as computing the frequencies distribution means using both neural networks, and SHAP can be used for a single neural network at a time
# FOR SHAP you need to find the predictions, not the loss, the frequencies are used not as predictions but as a way to compute the loss
# If we are interested in interpreting the first neural network, we should just get the output of the neural network (at this point also the deletion length should
# not be included, but I include them because they directly affect the mh score, which stands for the strength of the microhomology)

# From the supplementary material:
# This phi score represents the “strength” of the microhomology  corresponding to a particular MH deletion genotype. 
# It also trains MHless-NN which uses as input (deletion length) to directly output a phi score representing the “total strength” of all MH-independent activity for a particular deletion length.
def find_SHAP_values():
    francesco_rq_ans = pickle.load(open('outputaab/francesco_rq_ans.pkl', 'rb'))
    INP_train, INP_test, _, DEL_LENS_train, DEL_LENS_test = francesco_rq_ans
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
    nn_one_explainer = shap.Explainer(nn_one_predict_mh_score, sampled_nn_one_train_inputs)
    nn_one_shap_values = nn_one_explainer(nn_one_test_inputs)
    nn_one_shap_values.feature_names = ['MH length', 'GC content', 'Deletion length']

    
    pickle.dump(sampled_nn_one_train_inputs, open('outputaab/SHAP_nn_one_train_inputs.pkl', 'wb'))
    pickle.dump(nn_one_test_inputs, open('outputaab/SHAP_nn_one_test_inputs.pkl', 'wb'))
    pickle.dump(nn_one_shap_values, open('outputaab/SHAP_nn_one_shap_values.pkl', 'wb'))


    nn_two_train_inputs = del_feature_train.reshape((len(del_feature_train), 1))
    nn_two_test_inputs = del_feature_test.reshape((len(del_feature_test), 1))
    
    sampled_nn_two_train_inputs = shap.sample(nn_two_train_inputs, len(del_feature_train) // 8)

    # nn2 explainer
    # pass the network (with the trained parameters) to the explainer, and the background (to compute the mean for the SHAP values)
    nn_two_explainer = shap.Explainer(nn_two_predict_mh_less_score, sampled_nn_two_train_inputs)
    nn_two_shap_values = nn_two_explainer(nn_two_test_inputs)

    pickle.dump(sampled_nn_two_train_inputs, open('outputaab/SHAP_nn_two_train_inputs.pkl', 'wb'))
    pickle.dump(nn_two_test_inputs, open('outputaab/SHAP_nn_two_test_inputs.pkl', 'wb'))
    pickle.dump(nn_two_shap_values, open('outputaab/SHAP_nn_two_shap_values.pkl', 'wb'))


def save_SHAP_figures():
    shap_values_one = pickle.load(open('outputaab/SHAP_nn_one_shap_values.pkl', 'rb'))

    shap_values_one.feature_names = ['MH length', 'GC content', 'Deletion length']
    shap.plots.beeswarm(shap_values_one, show=False)
    plt.tight_layout()
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
    shap_values_two = pickle.load(open('outputaab/SHAP_nn_two_shap_values.pkl', 'rb'))
    shap_values_two.feature_names = ['Deletion length']
    shap.plots.beeswarm(shap_values_two, show=False)
    plt.tight_layout()
    plt.savefig('nn2_beeswarm.png')
    shap.plots.scatter(shap_values_two, show=False)
    plt.tight_layout()
    plt.savefig('nn2_scatter.png')

def find_SHAP_values_knn():
  model = pickle.load(open('model-mlbio/rate_model_v2.pkl', 'rb'))
  X = pickle.load(open('outputaab/X_knn.pkl', 'rb'))
  # Tranform the entropy into a precision score
  X[:, 4] = 1 - X[:, 4]
  explainer = shap.Explainer(model.predict, X)
  shap_values = explainer(X)
  pickle.dump(shap_values, open('outputaab/shap_values_knn.pkl', 'wb'))
  

def save_SHAP_figures_knn():
  shap_values = pickle.load(open('outputaab/shap_values_knn.pkl', 'rb'))
  shap_values.feature_names = ['-4G freq', '-4T freq', '-3A freq', '-3G freq', 'Precision score', 'DelScore (Total Phi)']
  shap.plots.beeswarm(shap_values, show=False)
  plt.tight_layout()
  #plt.savefig('knn_beeswarm.png')
  plt.show()
  shap.plots.scatter(shap_values, show=False)
  plt.tight_layout()
  #plt.savefig('knn_scatter.png')
  plt.show()
  shap.plots.bar(shap_values, show=False)
  plt.tight_layout()
  #plt.savefig('knn_bar.png')
  plt.show()
  shap.plots.heatmap(shap_values, show=False)
  plt.tight_layout()
  #plt.savefig('knn_heatmap.png')
  plt.show()
  shap.summary_plot(shap_values, show=False)
  plt.tight_layout()
  #plt.savefig('knn_summary_plot.png')
  plt.show()


# Not used
def computeLime():
    francesco_rq_ans = pickle.load(open('outputaab/francesco_rq_ans.pkl', 'rb'))
    INP_train, INP_test, DEL_LENS_train, DEL_LENS_test = francesco_rq_ans
    # Divide the whole INP_TRAIN in a n x 2 matrix, where 2 are the columns: GC content and MH length
    np.concatenate(INP_train).ravel().reshape((356642 // 2, 2))
    # Make an array of the DEL_LENS_train
    del_feature_train = np.concatenate(DEL_LENS_train).ravel()
    # Make a matrix of the two columns of INP_TRAIN and the DEL_LENS_TRAIN rows
    mh_features_train = np.concatenate(INP_train).ravel().reshape((len(del_feature_train), 2))
    del_feature_test = np.concatenate(DEL_LENS_test).ravel()

    # Concatenate mh_features (GC content and MH length) and del_feature
    X_train = np.c_[mh_features_train, del_feature_train]
    # Do the same for the test set
    mh_features_test = np.concatenate(INP_test).ravel().reshape((len(del_feature_test), 2))
    X_test = np.c_[mh_features_test, del_feature_test]
    
    pickle.dump(X_train, open('outputaab/lime_X_train.pkl', 'wb'))
    pickle.dump(X_test, open('outputaab/lime_X_test.pkl', 'wb'))

    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, feature_names=['MH len', 'GC frac', 'DEL len'], class_names=['output'], verbose=True, discretize_continuous=False, mode='regression')
    # Get explanation for a specific instance
    print("LIME explanation for the first instance of the test set:")
    print(X_test[0])
    exp = explainer.explain_instance(X_test[0], nn_one_predict_mh_score, num_features=3)
    # Print the explanation
    print(exp.as_list())
    
    # extract the feature importance and feature names from the explanation
    feature_importance = [x[1] for x in exp.as_list()]
    feature_names = [x[0] for x in exp.as_list()]

    # plot the feature importance values
    plt.figure(figsize=(8,6))
    plt.barh(range(len(feature_importance)), feature_importance, align='center', color='green')
    plt.yticks(range(len(feature_names)), feature_names, fontsize=10)
    plt.xlabel('Feature importance', fontsize=12)
    plt.title('LIME Feature Importance Summary', fontsize=14)
    plt.show()

if __name__ == '__main__':
    global nn_one_params, nn_two_params
    nn_one_params = pickle.load(open('model-mlbio/aae_nn.pkl', 'rb'))
    nn_two_params = pickle.load(open('model-mlbio/aae_nn2.pkl', 'rb'))

    #find_SHAP_values_knn()
    #save_SHAP_figures_knn()    

    #find_SHAP_values()
    #save_SHAP_figures()

    find_permutation_feature_importance()

    

