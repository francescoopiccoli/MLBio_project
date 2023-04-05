import sys
import os
root_folder = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_folder)
import numpy
import csv
import inDelphi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


test_targets = {}
with open('/Users/siradakaewchino/Desktop/Delft/Machine Learning in bioinformatics/MLBio_project/research-questions/test_targets_out.csv') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        test_targets[row[0]] = row[1]
    
highest_del_freq_list = []
lenght_of_highest_del_freq = []
MH_del_freq= []
MH_less_del_freq =[]
highest_ins_freq_list = []
phi_number = []
Precision_procent =[]
one_bp_ins_freq = []
frameshift_freq = []
frame_zero_freq = []
frame_one_freq = []
frame_two_freq = []
for target in test_targets.values():
    _, stats = inDelphi.predict(target, 27)
    highest_del_freq_list.append(stats["Highest del frequency"])
    highest_ins_freq_list.append(stats["Highest ins frequency"])
    lenght_of_highest_del_freq.append(stats["Expected indel length"])
    MH_del_freq.append(stats["MH del frequency"])
    MH_less_del_freq.append(stats["MHless del frequency"])
    Precision_procent.append(stats["Precision"])
    one_bp_ins_freq.append(stats["1-bp ins frequency"])
    frameshift_freq.append(stats["Frameshift frequency"])
    frame_zero_freq.append(stats["Frame +0 frequency"])
    frame_one_freq.append(stats['Frame +1 frequency'])
    frame_two_freq.append(stats['Frame +2 frequency'])
    phi_number.append(stats['Phi'])


# Create a DataFrame with the data list as a column
df = pd.DataFrame({'Highest deletion frequency': highest_del_freq_list, 'Lenght': lenght_of_highest_del_freq, 'MH deletion frequency': MH_del_freq,'MH less deletion frequency': MH_less_del_freq})

#'Phi':phi_number, 'Precision': Precision_procent,'1-bp ins frequency': one_bp_ins_freq,'Frame +0 freqency': frame_zero_freq,'Frame +1 frequency':frame_one_freq, 'Frame +2 frequency': frame_two_freq,'Highest ins frequency': highest_ins_freq_list
#print(df)
#PCA 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Choose the number of components for PCA
#calculates the minimum number of principal components you can obtain from the given dataset. 
#n_components = min(df.shape[0], df.shape[1])
n_components = 4
#print(n_components)

# Create a PCA instance and fit it on the standardized data
pca = PCA(n_components=n_components)
principal_components = pca.fit_transform(scaled_data)

# Get the loadings for each feature on the principal components
#loadings = pca.components_

# Create a DataFrame with the loadings
#loadings_df = pd.DataFrame(loadings, columns=df.columns, index=[f'PC{i+1}' for i in range(n_components)])


pc_columns = ['PC' + str(i + 1) for i in range(n_components)]
principal_df = pd.DataFrame(data=principal_components, columns=pc_columns)

#explained_variance_ratio = pca.explained_variance_ratio_
#cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# Categorize 'Highest deletion frequency' into bins
df['Highest deletion frequency category'] = pd.qcut(df['Highest deletion frequency'], 3, labels=['Low', 'Medium', 'High'])
#df['MH deletion freq category'] = pd.qcut(df['MH deletion frequency'], 3, labels=['Low', 'Medium', 'High'])

# Add this new column to the principal_df DataFrame
principal_df['Highest deletion frequency category'] = df['Highest deletion frequency category']
#principal_df['MH deletion freq category'] = df['MH deletion freq category']


# Create a scatter plot matrix (pair plot) using Seaborn
#sns.pairplot(principal_df, diag_kind='hist', markers='o', corner=False) #hue='Highest deletion frequency category'
#plt.show()


#Heatmap
#correlation_matrix = df.corr()
#plt.figure(figsize=(10, 8))
#sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5)
#plt.title("Correlation Matrix")
#plt.show()



# Create a biplot
def create_biplot(principal_components, pca, df, categories):
    plt.figure(figsize=(10, 8))
    
    # Define colors for each category
    colors = {'Low': 'blue', 'Medium': 'green', 'High': 'red'}

    # Scatter plot of the principal components
    for category, color in colors.items():
        idx = categories == category
        plt.scatter(principal_components[idx, 0], principal_components[idx, 1], c=color, label=category)

    # Add variable arrows
    for i, feature in enumerate(df.columns[:n_components]):
        plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
                  color='r', alpha=0.5, lw=2)
        plt.text(pca.components_[0, i], pca.components_[1, i], feature, fontsize=14)

    plt.xlabel(f"Principal Component 1 ({round(pca.explained_variance_ratio_[0] * 100, 2)}%)")
    plt.ylabel(f"Principal Component 2 ({round(pca.explained_variance_ratio_[1] * 100, 2)}%)")
    plt.title("PCA Biplot")
    plt.legend()
    plt.grid()
    plt.show()

# Call the create_biplot function with the additional 'categories' argument
create_biplot(principal_components, pca, df, df['Highest deletion frequency category'])




'''
# Create a scree plot for the explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Explained Variance Ratio')
plt.grid()
plt.show()

# Create a scree plot for the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.grid()
plt.show()
'''



