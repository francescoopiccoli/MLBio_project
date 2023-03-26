from inDelphiModel import inDelphi
import pandas as pd

inDelphi.init_model(celltype = 'mESC')


# Generate figure 1e
def generate_figure_1e(sequence, cutsite):


    # Get predictions and statistics
    pred_df, stats = inDelphi.predict(sequence, cutsite)

    # Adds a genotype colum
    pred_df = inDelphi.add_genotype_column(pred_df, stats)

    # Adds the mhless_genotypes
    pred_df = inDelphi.add_mhless_genotypes(pred_df, stats)
    
    print(pred_df.sort_values(by='Predicted frequency', ascending=False).head(20))
    print(stats)
    pd.reset_option('display.max_rows')



def generate_figure_3f(predictions):

    pass
  # Get highest ins rate for each sequence

  # Get highest deletion rate for each sequence

  # Plot them

if __name__ == '__main__':
    left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
    right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
    seq = left_seq + right_seq
    cutsite = len(left_seq)

    generate_figure_1e(seq, cutsite)