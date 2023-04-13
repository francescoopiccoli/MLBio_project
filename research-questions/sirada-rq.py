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
'''
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

        exp_data = exp_data[((exp_data["Indel"].str.startswith("1+")) & (exp_data["Type"] == "INSERTION")) | ((exp_data["Type"] == "DELETION") & (exp_data["Size"]))]  #<= 28

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
#print(df)

# Save the DataFrame to a CSV file
#clear
df.to_csv('output_sirada.csv', index=False)
'''

df = pd.read_csv('/Users/siradakaewchino/Desktop/Delft/Machine Learning in bioinformatics/MLBio_project/research-questions/output_sirada.csv')

# Normalize the data
normalized_df = (df - df.min()) / (df.max() - df.min())

#PCA 
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Choose the number of components for PCA
#calculates the minimum number of principal components you can obtain from the given dataset. 
# n_components = min(df.shape[0], df.shape[1])
n_components = 4


# Create a PCA instance and fit it on the standardized data
pca = PCA(n_components)
principal_components = pca.fit_transform(normalized_df)

# Create a new DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=[f'Principal Component {i+1}' for i in range(principal_components.shape[1])])

# Create a scatter plot matrix (pair plot) using Seaborn
sns.pairplot(principal_df)
plt.show()

# Calculate explained variance
explained_variance = pca.explained_variance_ratio_ * 100


'''
# Create a DataFrame with the principal components
principal_df = pd.DataFrame(data=principal_components, columns=['Principal Component 1', 'Principal Component 2'])

# Plot the scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Scatter Plot')
plt.show()
'''

'''
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

'''

# Create a scree plot for the explained variance ratio
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot - Explained Variance Ratio')
plt.grid()
plt.show()

'''
# Create a scree plot for the cumulative explained variance
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance, marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot - Cumulative Explained Variance')
plt.grid()
plt.show()
'''

