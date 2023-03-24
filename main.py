import sys
import os
import numpy
import inDelphi
import pandas as pd

left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
seq = left_seq + right_seq
cutsite = len(left_seq)

pred_df, stats = inDelphi.predict(seq, cutsite)

pd.options.display.float_format = '{:.2f}'.format
print(pred_df.tail(4))
pred_df.sort_values(by = 'Predicted frequency', ascending = False, inplace = True)
print(pred_df.head(10))
