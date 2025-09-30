#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 14:38:49 2024

@author: tshan@urmc-sh.rochester.edu
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from expyfun.io import write_hdf5, read_hdf5, read_tab
from mtrf.model import TRF
from mtrf.stats import nested_crossval
import mne
#import cupy
import matplotlib.pyplot as plt
import os
import json
# %% Parameters
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_max = 27
n_epoch_total = 132
n_epoch_ori = 66
n_epoch_chimera = 66
# EEG param
eeg_n_channel = 2 # total channel of ABR
eeg_fs = 10000 # eeg sampling frequency
eeg_fs_down = 125
eeg_n_channel = 32
eeg_f_hp = 1 # high pass cutoff


channel_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9',
                 'CP5', 'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8',
                 'TP10', 'CP6', 'CP2', 'Cz', 'C4', 'T8', 'FT10', 'FC6', 'FC2',
                  'F4', 'F8', 'Fp2']
n_channel_per_group = 4
channel_groups = [channel_names[i:i+n_channel_per_group] for i in range(0, len(channel_names), n_channel_per_group)] 
ref_channels=["TP9", "TP10"]

# %% File paths
exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA EXT/Chimera/'

# %% Get EEG data, save into one file
subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']
# %%
fname = []
category = []
set_num = []
prefer = []
subj = []
# Load set pair value data
set_dic = read_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')
for subject in subject_list:
    # import EEG
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    for file in os.listdir(data_root):
        if file.endswith(".tab"):
            dataframe_path = data_root +"/"+ file
    tab_data = read_tab(dataframe_path)
    tab_data_trial_param = [i['Trial Params'] for i in tab_data]

    for ti in range(len(tab_data_trial_param)):
        subj += [subject]
        data_dict = json.loads(tab_data_trial_param[ti][0][0].replace("'",'"'))
        fname += [data_dict['trial_id'][:-4]]
        if data_dict['type_number']==0:
            cat = 'original'
        else:
            cat = 'chimeric'
        category += [cat]
        prefer += [data_dict['preferenc']]
        set_num += [set_dic[fname[-1]]]

df = pd.DataFrame(list(zip(subj, fname, category, set_num, prefer)),
                  columns=['subject','fname','category','group','preference'])

df.to_pickle('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/preference.pkl')

# %%
import statsmodels.formula.api as smf
import scipy
df["random_eff"] = df["subject"].astype(str) + "+" + df["group"].astype(int).astype(str)



r_lm = smf.mixedlm("preference ~ category", df, groups="random_eff").fit()
print(r_lm.summary())

# %% Plot
figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/paper_figs/'
orig_mean = df[df['category']=='original']['preference'].mean()
orig_sem = df[df['category']=='original']['preference'].std()/np.sqrt(len(df[df['category']=='original']))
chimera_mean = df[df['category']=='chimeric']['preference'].mean()
chimera_sem = df[df['category']=='chimeric']['preference'].std()/np.sqrt(len(df[df['category']=='chimeric']))

dpi=300
plt.figure()
barWidth = 0.5
br1 = np.arange(0.5,2.5)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.bar(br1[0], orig_mean, color='C0', width=barWidth, label='Original',zorder=3)
plt.bar(br1[1], chimera_mean, color='C1',label='Chimeric', width=barWidth,zorder=3)
plt.errorbar([br1[0],br1[1]], y=[orig_mean,chimera_mean], yerr=[orig_sem,chimera_sem],linestyle="", capsize=2, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus Category')
plt.ylabel("Preference Rating")
plt.xticks([r + barWidth for r in range(2)], ['Original', 'Chimera'])
plt.ylim(0, 5)
# plt.xlim(-0.25, 1.5)

#lg = plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.savefig(figure_path+'predict_real_corr_class_200ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
#plt.savefig(figure_path+'predict_real_corr_class_15ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
plt.tight_layout()
plt.savefig(figure_path+'preference_dist.png', dpi=dpi, format='png')