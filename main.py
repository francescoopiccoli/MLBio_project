import sys
import os
import numpy
import csv
import inDelphi
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

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


kde = gaussian_kde(highest_del_freq_list)
# Make a histogram of the frequencies
fig, ax = plt.subplots(1, 1)
x = np.linspace(0, 100, 1000)
ax.hist(highest_del_freq_list, bins = 100, density=True, color = 'red', alpha = 0.5)
ax.plot(x, kde(x), linewidth=2, color='c')
# Adds labels to the figure
ax.set_xlabel('Most frequent deletion genotype (%)')
ax.set_ylabel('Number of gRNAs')

# Save figure to output folder
fig.savefig("output/figure_3f_deletions.png",dpi=300, bbox_inches = "tight")

kde2 = gaussian_kde(highest_ins_freq_list)
fig2, ax2 = plt.subplots(1, 1)
x = np.linspace(0, 30, 150)
ax2.hist(highest_ins_freq_list, bins = 15, density=True, color = 'blue', alpha = 0.5)
ax2.plot(x, kde2(x), linewidth=2, color='k')
# Adds labels to the figure
ax2.set_xlabel('Most frequent insertion genotype (%)')
ax2.set_ylabel('Number of gRNAs')

# Save figure to output folder
fig2.savefig("output/figure_3f_insertions.png",dpi=300, bbox_inches = "tight")

# pred_df, stats = inDelphi.predict(seq, cutsite)

"""pd.options.display.float_format = '{:.2f}'.format
print(pred_df.tail(4))
pred_df.sort_values(by = 'Predicted frequency', ascending = False, inplace = True)
print(pred_df.head(10))"""
