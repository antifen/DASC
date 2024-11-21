import os

import pandas as pd
result_dir = '/root/cls/local_data/results/ganzhou'
experient_2nd = '2nd_fp22_penalty'
experient_3nd_pos ='3rd_hard_clus_pos_22penalty'
experient_3nd_neg = '3rd_hard_clus_neg_22penalty'

df = pd.read_csv(os.path.join(os.path.join(result_dir,experient_2nd),'sample2.csv'), sep=',')

main_df = df
sub_df0 = pd.read_csv(os.path.join(os.path.join(result_dir,experient_3nd_pos),'hard_1.csv'))
sub_df1 = pd.read_csv(os.path.join(os.path.join(result_dir,experient_3nd_neg),'hard_0.csv'))
sub_df0 = sub_df0[sub_df0['correct'] / sub_df0['sample_count'] >= 0.5]
sub_df1 = sub_df1[sub_df1['correct'] / sub_df1['sample_count'] >= 0.5]
sub_df = pd.concat([sub_df0, sub_df1], ignore_index=True)



df['type'] = -1
df.loc[df['Correction Flags'] != 0, 'type'] = 0
df.loc[(df['Correction Flags'] == 0) & (df['Hard Flags'] != 0), 'type'] = 1
df.loc[(df['Correction Flags'] == 0) & (df['Hard Flags'] == 0), 'type'] = 2

sub_df.set_index('File Name', inplace=True)


def get_hard_labels(file_name):
    if file_name in sub_df.index:
        return sub_df.loc[file_name, 'labels']
    else:
        return -1

main_df['hard_labels'] = main_df['File Name'].apply(get_hard_labels)

max_hard_label = main_df['hard_labels'].max()


main_df.loc[main_df['type'] == 2, 'hard_labels'] = max_hard_label + 1
main_df.loc[main_df['type'] == 0, 'hard_labels'] = -1


main_df = main_df[main_df['hard_labels'] != -1]
count = len(main_df[(main_df['labels'] ==1)])
print('NEG',count)
count = len(main_df[(main_df['labels'] ==0)])
print('POS',count)
main_df.to_csv(os.path.join(os.path.join(result_dir,experient_2nd),'SCL.csv'), index=False)