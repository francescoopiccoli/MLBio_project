from inDelphiModel import inDelphi

inDelphi.init_model(celltype = 'mESC')

left_seq = 'AGAATCGCCCGCGGTCCATCCTTTATCAGCGGGAATTCAAGCGCACCAGCCAGAGGTGTA'
right_seq = 'CCGTGGACGTGAGAAAGAAGAAACATAATATTCGCACTAGATCCATCCCCATACCTGACC'
seq = left_seq + right_seq
cutsite = len(left_seq)

pred_df, stats = inDelphi.predict(seq, cutsite)