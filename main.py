import sys
import os
import numpy
import csv
from inDelphi import find_observed_freqs
import pandas as pd
# From Boris: load test targets
test_targets = {}
with open('output_test/test_targets.csv') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        test_targets[row[0]] = row[1]

left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
seq = left_seq + right_seq
cutsite = len(left_seq)
find_observed_freqs(test_targets)
"""
pred_df, stats = inDelphi.predict(seq, cutsite)

pd.options.display.float_format = '{:.2f}'.format
print(pred_df.tail(4))
pred_df.sort_values(by = 'Predicted frequency', ascending = False, inplace = True)
print(pred_df.head(10))"""