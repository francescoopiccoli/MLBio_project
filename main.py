import sys
import os
import numpy
import csv
import inDelphi
import pandas as pd
import matplotlib.pyplot as plt
# From Boris: load test targets
test_targets = {}
with open('output_test/test_targets.csv') as file:
    csvreader = csv.reader(file)
    for row in csvreader:
        test_targets[row[0]] = row[1]

"""left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
seq = left_seq + right_seq
cutsite = len(left_seq)"""

#           GTTCTACAGATTGCTTG|TAC
# AACACTCCCTGTTCTACAGATTGCTTG|TACTGGTGAACAACATATTGGTATTCAA
# iterate over all test targets
highest_del_freq_list = []
highest_ins_freq_list = []
for target in test_targets.values():
    _, stats = inDelphi.predict(target, 27)
    highest_del_freq_list.append(stats["Highest del frequency"])
    highest_ins_freq_list.append(stats["Highest ins frequency"])


# Make a histogram of the frequencies
plt.hist(highest_del_freq_list, bins = 100)
plt.show()

# pred_df, stats = inDelphi.predict(seq, cutsite)

"""pd.options.display.float_format = '{:.2f}'.format
print(pred_df.tail(4))
pred_df.sort_values(by = 'Predicted frequency', ascending = False, inplace = True)
print(pred_df.head(10))"""
