import pickle
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import _tree
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error

# Get the precision-X score for each sequence
def parse_input_data(data):
  # We care about deletions (MH and MH-less) for the neural networks.

  deletions_data = data[data['Type'] == 'DELETION'].reset_index()
  exps = []
  exp_precision_x = []

  # To make this run in a short time, take only the first n elements (i.e. [:n])
  exps = deletions_data['Sample_Name'].unique()

  for exp in exps:
    exp_data = deletions_data[deletions_data['Sample_Name'] == exp]
    dl_freq_data = exp_data[exp_data['Size'] <= 28]
    total_deletion_events = sum(dl_freq_data['countEvents'])

    precision_x = float('-inf')
    for index, row in dl_freq_data.iterrows():
       dl_freq = row['countEvents'] / total_deletion_events
       if dl_freq > precision_x:
          precision_x = dl_freq
    exp_precision_x.append(precision_x)

  return [exps, exp_precision_x]

# Used to get decision tree rules
# From https://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree, by pplonski
def get_rules(tree, feature_names, class_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []
    
    def recurse(node, path, paths):
        
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, 3)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, 3)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:
            path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]
            
    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]
    
    rules = []
    for path in paths:
        rule = "if "
        
        for p in path[:-1]:
            if rule != "if ":
                rule += " and "
            rule += str(p)
        rule += " then "
        if class_names is None:
            rule += "response: "+str(np.round(path[-1][0][0][0],3))
        else:
            classes = path[-1][0][0]
            l = np.argmax(classes)
            rule += f"class: {class_names[l]} (proba: {np.round(100.0*classes[l]/np.sum(classes),2)}%)"
        rule += f" | based on {path[-1][1]:,} samples"
        rules += [rule]
        
    return rules

# Main, load data and preprocess data, train model, run/evaluate model
if __name__ == '__main__':
    shap.initjs() 
    # Load and parse input data
    inp_dir = '../input/'
    master_data = pickle.load(open(inp_dir + 'inDelphi_counts_and_deletion_features.pkl', 'rb'))

    # counts: Contains a dataframe detailing the observed counts for each repair outcome (insertions and deletions) for every target sequence. The “fraction” column can be ignored.
    counts = master_data['counts'].drop('fraction', axis=1)

    # del_features: contains a dataframe detailing the deletion length, homology length, and homology GC content, for each deletion-type repair outcome for every target sequence.
    del_features = master_data['del_features']

    # merged counts and del_features
    data = pd.concat((counts, del_features), axis=1)
    
    # For each sequence find the highest repair outcome and store that percentage in a new df with two columns [sequence, percision X-score]
    [exps, precision_x] = parse_input_data(data)

    print(len(exps))
    for exp in precision_x:
        print(exp)

    # Add the three nucleotides left to the cutesite as well as right to the cutsite to a list (for each sequence)
    pre_X = []
    for exp in exps:
       pre_X.append(list(exp[(len(exp) - 6):(len(exp) - 3)] + exp[(len(exp) - 3):]))

    # One hot encode our feature vector
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(pre_X)

    X_encoded = enc.transform(pre_X).toarray()

    feature_names = ['-3bp', '-2bp', '-1bp', '1bp', '2bp', '3bp']
    new_feature_names = enc.get_feature_names_out(feature_names)
    print(new_feature_names)
    X_encoded = pd.DataFrame(X_encoded, columns= new_feature_names)

    df_Y = pd.DataFrame({'precision_x': np.array(precision_x)})
    df_Y['precision_x'] = pd.cut(df_Y['precision_x'], bins=10)
    
    Y = []
    for index, row in df_Y.iterrows():
      Y.append(str(row['precision_x']))

    print(df_Y)

    Y = np.array(precision_x)

    # Split between training and test set
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, Y, 
                                                    test_size=0.1, 
                                                    random_state=1)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}\n")
    print(f"Test features shape: {X_test.shape}")
    print(f"Test target shape: {y_test.shape}")

    # Train a decision tree model using the sequence as input (as a one-hot encoded array) and the percision-X score as target vector
    clf = tree.DecisionTreeRegressor(max_depth=4, criterion="friedman_mse")
    clf = clf.fit(X_train, y_train)

    # Get MSE of model using test set
    print("MSE: " , np.round(mean_squared_error(y_test, clf.predict(X_test)), 2))
  
    # Run TreeShap on the decision tree to learn more about patterns with the sequence that cause it to give a high percision-X score
    explainer = shap.TreeExplainer(clf)
    shap_test = explainer(X_test)

    #shap.plots.waterfall(shap_test[0])
    shap.summary_plot(shap_test)

    print(f"Shap values length: {len(shap_test)}\n")
    print(f"Sample shap value:\n{shap_test[0]}")
    plt.figure(figsize=(24,32)) 
    tree.plot_tree(clf, feature_names=new_feature_names, fontsize=4)

    plt.show()

    # Print out decision tree rules
    rules = get_rules(clf, new_feature_names, None)
    for r in rules:
      print(r)
