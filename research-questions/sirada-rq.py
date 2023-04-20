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
import pickle


# Find the observed frequencies for the test targets
def find_observed_freqs(test_targets):
    master_data = pickle.load(open('/Users/siradakaewchino/Desktop/Delft/Machine Learning in bioinformatics/MLBio_project/input/inDelphi_counts_and_deletion_features.pkl', 'rb'))
    counts = master_data['counts'].drop('fraction', axis=1)
    del_features = master_data['del_features']
    # merged counts and del_features
    data = pd.concat((counts, del_features), axis=1).reset_index()

    #highest_deletion_info = []
    
    highest_deletion_freqs = []
    deletion_lengths = []
    MH_deletion_freqs = []
    MHless_deletion_freq = []


    for exp in test_targets.keys():
        exp_data = data[data["Sample_Name"].str.endswith(exp)]
        exp_data.dropna(subset=["countEvents"], inplace=True)

        exp_data = exp_data[((exp_data["Indel"].str.startswith("1+")) & (exp_data["Type"] == "INSERTION")) | ((exp_data["Type"] == "DELETION") & (exp_data["Size"]<= 28))]  #<= 28

        counts = exp_data.groupby("Indel")["countEvents"].sum()
        total_events = exp_data["countEvents"].sum()
        freqs = counts / total_events * 100

        exp_data["Frequencies (%)"] = exp_data["Indel"].map(freqs)
        exp_data.sort_values(by="Frequencies (%)", ascending=False, inplace=True)

        deletion_data = exp_data[exp_data["Type"] == "DELETION"]
        if not deletion_data.empty:
            max_deletion_freq = deletion_data["Frequencies (%)"].max()
            max_deletion_row = deletion_data[deletion_data["Frequencies (%)"] == max_deletion_freq].iloc[0]
            deletion_length = max_deletion_row["Size"]
            
            
            # Filter the data for MH deletions and MH less deletions
            mh_deletion_data = deletion_data[deletion_data["homologyLength"] > 0]
            non_mh_deletion_data = deletion_data[deletion_data["homologyLength"] == 0]

        
            # Correctly identify mh-less deletions
            #mhlessQuery = '("Type" == "DELETION") & (1 <= homologyLength <= 60)'
            #non_mh_deletion_data.loc[non_mh_deletion_data.query(mhlessQuery).index, 'Type'] = "mh-less del"

        if not mh_deletion_data.empty:
            max_mh_deletion_freq = mh_deletion_data["Frequencies (%)"].max()
        else:
            max_mh_deletion_freq = 0

        if not non_mh_deletion_data.empty:
            max_non_mh_deletion_freq = non_mh_deletion_data["Frequencies (%)"].max()  
        else:
            max_non_mh_deletion_freq = 0

            #highest_deletion_info.append((max_deletion_freq, deletion_length, max_mh_deletion_freq, max_non_mh_deletion_length))  # Append the highest deletion frequency, length, MH deletion frequency, and non-MH deletion length as a tuple
            
        highest_deletion_freqs.append(max_deletion_freq)
        deletion_lengths.append(deletion_length)
        MH_deletion_freqs.append(max_mh_deletion_freq)
        MHless_deletion_freq.append(max_non_mh_deletion_freq)

    #return highest_deletion_info  # Return the list of tuples

    return (highest_deletion_freqs, deletion_lengths,
            MH_deletion_freqs, MHless_deletion_freq)


if __name__ == '__main__':

    # Code for loading test_targets and calling the find_observed_freqs() function remains unchanged
    # Provide test_targets manually for this example
    test_targets = {}
    with open('/Users/siradakaewchino/Desktop/Delft/Machine Learning in bioinformatics/MLBio_project/research-questions/test_targets_out.csv') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            test_targets[row[0]] = row[1]
    # Call find_observed_freqs() and store the result in a variable
    highest_deletion_info = find_observed_freqs(test_targets)

# Call find_observed_freqs() and store the result in variables
highest_deletion_freqs, deletion_lengths, MH_deletion_freqs, MHless_deletion_freq = find_observed_freqs(test_targets)

# Create a DataFrame with the frequency values
data = {'Highest Deletion Frequencies': highest_deletion_freqs,
        'Deletion Lengths': deletion_lengths,
        'Highest MH Deletion Frequencies': MH_deletion_freqs,
        'Highest MHless Deletion Frequencies': MHless_deletion_freq}

df = pd.DataFrame(data)

# Print the DataFrame
print(df)

# Save the DataFrame to a CSV file
#clear
df.to_csv('output_sirada.csv', index=False)


df = pd.read_csv('/Users/siradakaewchino/Desktop/Delft/Machine Learning in bioinformatics/MLBio_project/research-questions/output_sirada.csv')


#PCA 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Choose the number of components for PCA
#calculates the minimum number of principal components you can obtain from the given dataset. 
n_components = 4


# Create a PCA instance and fit it on the standardized data
pca = PCA(n_components)
principal_components = pca.fit_transform(df)

# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=[f'Principal Component {i+1}' for i in range(principal_components.shape[1])])

# Plotting

# Create a scatter plot matrix (pair plot) using Seaborn
sns.pairplot(principal_df)
plt.show()

# Loading plot
loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

def biplot(score, loadings, columns, pc_x=0, pc_y=1):
    plt.figure(figsize=(10, 10))
    
    # Plot the samples (projected onto the principal components)
    plt.scatter(score[:, pc_x], score[:, pc_y], alpha=0.5)
    
    # Plot the loadings (original variables as vectors)
    for i, txt in enumerate(columns):
        plt.arrow(0, 0, loadings[i, pc_x], loadings[i, pc_y], color='r', head_width=0.05, head_length=0.1, linewidth=1.5)
        plt.annotate(txt, (loadings[i, pc_x], loadings[i, pc_y]), fontsize=12, fontweight='bold', color='r')
        
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel(f'Principal Component {pc_x + 1}')
    plt.ylabel(f'Principal Component {pc_y + 1}')
    plt.title('PCA Biplot')
    
    # Set axis limits based on loadings and scores
    plt.xlim(min(score[:, pc_x].min(), loadings[:, pc_x].min()) - 0.1, max(score[:, pc_x].max(), loadings[:, pc_x].max()) + 0.1)
    plt.ylim(min(score[:, pc_y].min(), loadings[:, pc_y].min()) - 0.1, max(score[:, pc_y].max(), loadings[:, pc_y].max()) + 0.1)
    
    plt.grid()
    plt.show()
    
biplot(principal_components, loadings, df.columns)


# Create a scree plot for the explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
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


