import sys
import os
import numpy
import inDelphi

left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
seq = left_seq + right_seq
cutsite = len(left_seq)

pred_df, stats = inDelphi.predict(seq, cutsite)
print(pred_df)
print(stats)