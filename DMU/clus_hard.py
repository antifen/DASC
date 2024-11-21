
import csv
import pandas as pd
import os


result_dir = '/root/cls/local_data/results/ganzhou/2nd_fp22_penalty'
path_csv = '/root/data/tupac/tog_csv_reverse'
harc_path_csv = '/root/data/icpr2014/hard/tog_csv_reverse_hard'
hard_images_path = r'/root/data/icpr2014/hard/only_hard_images'
sampe_csv = os.path.join(result_dir,'sample2.csv')

df = pd.read_csv(sampe_csv, sep=',')
#0 NOISE 1 HARD 3 CLEAN
df['type'] = -1
df.loc[df['Correction Flags'] != 0, 'type'] = 0
df.loc[(df['Correction Flags'] == 0) & (df['Hard Flags'] != 0), 'type'] = 1
df.loc[(df['Correction Flags'] == 0) & (df['Hard Flags'] == 0), 'type'] = 2

df['forget_rate'] = df.apply(lambda x: 1 if x['learning'] == 0 else x['forget'] / (x['forget'] + x['learning']), axis=1)
df['scd_rate'] = df.apply(lambda x: (x['scd']+1e-6) / (x['forget'] + x['learning'] - 20), axis=1)
df = df[df['forget_rate'] != 1.0]
gmm_df = df
gmm_df = gmm_df[(gmm_df['type'] != 2 ) & ( gmm_df['type'] != 0)]
gmm_hard(gmm_df)


