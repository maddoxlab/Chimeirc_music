#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:10:03 2024

@author: tshan@urmc-sh.rochester.edu
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from mne.filter import resample
from expyfun.io import read_wav, write_hdf5, read_hdf5
from mtrf.model import TRF, load_sample_data
from mtrf.stats import crossval, nested_crossval
from statsmodels.stats import multitest
import mne
#import cupy
import pickle
import matplotlib.pyplot as plt
import os
# %%
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
regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
# %%
downbeat_df = pd.read_pickle(regressor_path+'downbeat_trf_reg.pkl')
ori_downbeat_df = downbeat_df[downbeat_df['type']==0]
ori_downbeat_df.reset_index(inplace=True)
exp_df = pd.read_pickle(regressor_path+'expectation_downbeat_seq_ltm_b16.pkl')
note_dur = []
downbeat = []
for i in range(len(ori_downbeat_df)):
    f_name = ori_downbeat_df.loc[i, 'name']
    f_name_ori = f_name[:-4]
    note_ind = np.where(ori_downbeat_df.iloc[i]['onset']==1)[0]
    last = len(ori_downbeat_df.iloc[i]['onset'])-1
    
    exp_ind = exp_df[exp_df['name']==f_name].index.values.astype(int)[0]
    downbeat_both_temp = exp_df.loc[exp_ind, 'downbeat_both']
    
    for ni in range(len(note_ind)):
        if (ni == 0) or (ni == len(note_ind)-1):
            pass
        else:
            note_dur += [(note_ind[ni]-1-note_ind[ni-1])/eeg_fs_down]
            downbeat += [downbeat_both_temp[ni]]

note_dur = np.array(note_dur)
downbeat = np.array(downbeat)

plt.hist(note_dur[np.where(downbeat==0)[0]], bins=np.arange(0,2.5,0.05),label="other", density=True, histtype='step')
plt.hist(note_dur[np.where(downbeat==1)[0]], bins=np.arange(0,2.5,0.05),label="downbeat",density=True, histtype='step')



