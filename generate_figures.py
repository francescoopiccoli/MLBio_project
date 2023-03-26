from inDelphiModel import inDelphi
import pandas as pd
import numpy as np

# Generate figure 1e
def generate_figure_1e(sequence, cutsite):


    # Get predictions and statistics
    pred_df, stats = inDelphi.predict(sequence, cutsite)

    # Adds the mhless_genotypes
    pred_df = inDelphi.add_mhless_genotypes(pred_df, stats)

    # Adds a genotype colum
    pred_df = inDelphi.add_genotype_column(pred_df, stats)

    # Need because of python version
    pred_df = pred_df.rename(columns={'Genotype position': 'Genotype_position'})

    # Correctly identify mh-less deletions
    query = '(Category == \'del\') & (1 <= Length <= 60) & (0 <= Genotype_position <= Length)'
    pred_df.loc[pred_df.query(query).index,'Category'] = "mh-less del"

    # Print data frame
    print(pred_df.sort_values(by='Predicted frequency', ascending=False).head(20)[['Genotype', 'Category', 'Predicted frequency']].to_string(index=False))
    # print(stats)

    pd.reset_option('display.max_rows')



def generate_figure_3f(predictions):
  pass
  # Get highest ins rate for each sequence

  # Get highest deletion rate for each sequence

  # Plot them

if __name__ == '__main__':

    # Init inDelphi with mouse cell dataset
    inDelphi.init_model(celltype = 'mESC')

    left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
    right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
    seq = left_seq + right_seq
    cutsite = len(left_seq)

    # Generate figure 1e
    generate_figure_1e(seq, cutsite)