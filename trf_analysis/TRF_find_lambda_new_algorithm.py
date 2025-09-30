#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 12:01:19 2024

@author: tshan
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from expyfun.io import write_hdf5, read_hdf5
from mtrf.model import TRF
from mtrf.matrices import lag_matrix
from mtrf.stats import nested_crossval
import mne
#import cupy
import matplotlib.pyplot as plt
import os

mne.utils.set_config('MNE_USE_CUDA', 'true')  
mne.cuda.init_cuda(verbose=True)

# %% Filtering
def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass_filtfilt(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return 
# %% Z score func
def z_score_df(df, feature):
    all_features = np.concatenate(df[feature].values)
    global_mean = np.mean(all_features)
    global_std = np.std(all_features)
    return  df[feature].apply(lambda x: (x-global_mean)/global_std)
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
exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA/Chimera/'
# %% subject list
subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

trf_ori_AM_reg = []
trf_ori_LTM_reg = []
trf_ori_STM_reg = []
trf_ori_A_reg = []
trf_chimera_AM_reg = []
trf_chimera_LTM_reg = []
trf_chimera_STM_reg = []
trf_chimera_A_reg = []

# %%
do_zscore = True
# %%
##### Get matrix for X
onset_cut = 1
exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA/Chimera/'
regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
# Load set pair value data
set_dic = read_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')

t_min = -0.1
t_max = 0.7
# Load regressors
# env_df = pd.read_pickle(regressor_path+'env.pkl')
flux_df = pd.read_pickle(regressor_path+'flux.pkl')
onset_df = pd.read_pickle(regressor_path+'onset.pkl')

# expectation LTM
ltm_exp_df = pd.read_pickle(regressor_path+'expectation_ds_ltm_nb_pitch_ioi.pkl')
# expectation STM
stm_exp_df = pd.read_pickle(regressor_path+'expectation_ds_stm_nb_pitch_ioi.pkl')
# expectation both
both_exp_df = pd.read_pickle(regressor_path+'expectation_ds_both_nb_pitch_ioi.pkl')

# Normalize regressors zscore
if do_zscore:
    flux_df['flux_ds_z'] = z_score_df(flux_df,'flux_ds')
    onset_df['onset_ds_z'] = z_score_df(onset_df,'onset_ds')
    ltm_exp_df['sp_ds_z'] = z_score_df(ltm_exp_df,'sp_ds')
    ltm_exp_df['hp_ds_z'] = z_score_df(ltm_exp_df,'hp_ds')
    ltm_exp_df['so_ds_z'] = z_score_df(ltm_exp_df,'so_ds')
    ltm_exp_df['ho_ds_z'] = z_score_df(ltm_exp_df,'ho_ds')
    stm_exp_df['sp_ds_z'] = z_score_df(stm_exp_df,'sp_ds')
    stm_exp_df['hp_ds_z'] = z_score_df(stm_exp_df,'hp_ds')
    stm_exp_df['so_ds_z'] = z_score_df(stm_exp_df,'so_ds')
    stm_exp_df['ho_ds_z'] = z_score_df(stm_exp_df,'ho_ds')
    both_exp_df['sp_ds_z'] = z_score_df(both_exp_df,'sp_ds')
    both_exp_df['hp_ds_z'] = z_score_df(both_exp_df,'hp_ds')
    both_exp_df['so_ds_z'] = z_score_df(both_exp_df,'so_ds')
    both_exp_df['ho_ds_z'] = z_score_df(both_exp_df,'ho_ds')


eeg_data = read_hdf5(exp_path+'subject'+'_001/'+'chimera_001'+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
all_ori_epochs_df = eeg_data['ori_epochs_df']
all_chimera_epochs_df = eeg_data['chimera_epochs_df']

# Trim epochs and concatenate epochs
x_in_ori_trim_LTM = [] 
x_in_ori_trim_STM = []
x_in_ori_trim_both = []
x_in_ori_trim_AM = []
x_in_ori_trim_A =[]

x_out_ori_trim = []
f_name_ori = []
set_num_ori = []

for ei in range(n_epoch_ori):
    f_name = all_ori_epochs_df.loc[ei, 'name']
    f_name_ori += [f_name[:-4]]
    set_num_ori += [set_dic[f_name_ori[-1]]]
    # cut onset for both regressor and eeg data
    # Get x_in
    # Flux
    flux_ind = flux_df[flux_df['name']==f_name].index.values.astype(int)[0]
    epoch_len = np.floor(flux_df.loc[flux_ind, 'duration']*eeg_fs_down)
    flux_x_in_temp = flux_df.loc[flux_ind, 'flux_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Onset
    onset_ind = onset_df[onset_df['name']==f_name].index.values.astype(int)[0]
    #epoch_len = np.floor(onset_df.loc[onset_ind, 'duration']*eeg_fs_down)
    onset_x_in_temp = onset_df.loc[onset_ind, 'onset_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation LTM
    ltm_exp_ind = ltm_exp_df[ltm_exp_df['name']==f_name].index.values.astype(int)[0]
    ltm_sp_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_hp_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_so_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_ho_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation STM
    stm_exp_ind = stm_exp_df[stm_exp_df['name']==f_name].index.values.astype(int)[0]
    stm_sp_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_hp_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_so_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_ho_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation both
    both_exp_ind = both_exp_df[both_exp_df['name']==f_name].index.values.astype(int)[0]
    both_sp_x_in_temp = both_exp_df.loc[both_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_hp_x_in_temp = both_exp_df.loc[both_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_so_x_in_temp = both_exp_df.loc[both_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_ho_x_in_temp = both_exp_df.loc[both_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    

        
    x_in_temp_A = np.array([flux_x_in_temp,onset_x_in_temp])
    x_in_temp_LTM = np.array([flux_x_in_temp,onset_x_in_temp,ltm_sp_x_in_temp,
                          ltm_hp_x_in_temp,ltm_so_x_in_temp,ltm_ho_x_in_temp])
    x_in_temp_STM = np.array([flux_x_in_temp,onset_x_in_temp,stm_sp_x_in_temp,
                          stm_hp_x_in_temp,stm_so_x_in_temp,stm_ho_x_in_temp])
    x_in_temp_both = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,
                          both_hp_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
    x_in_temp_AM = np.array([flux_x_in_temp,onset_x_in_temp,ltm_sp_x_in_temp,
                          ltm_hp_x_in_temp,ltm_so_x_in_temp,ltm_ho_x_in_temp,
                          stm_sp_x_in_temp,stm_hp_x_in_temp,stm_so_x_in_temp,stm_ho_x_in_temp])
    
    # Transpose to match mTRF toolbox dimension
    x_in_ori_trim_A += [x_in_temp_A.T]
    x_in_ori_trim_LTM += [x_in_temp_LTM.T]
    x_in_ori_trim_STM += [x_in_temp_STM.T]
    x_in_ori_trim_both += [x_in_temp_both.T]
    x_in_ori_trim_AM += [x_in_temp_AM.T]

# Trim epochs and concatenate epochs
x_in_chimera_trim_LTM = [] 
x_in_chimera_trim_STM = []
x_in_chimera_trim_both = []
x_in_chimera_trim_AM = []
x_in_chimera_trim_A = []
x_out_chimera_trim = []
f_name_chimera = []
set_num_chimera = []

for ei in range(n_epoch_chimera):
    f_name = all_chimera_epochs_df.loc[ei, 'name']
    f_name_chimera += [f_name[:-4]]
    set_num_chimera += [set_dic[f_name_chimera[-1]]]
    # cut onset for both regressor and eeg data
    # Get x_in
    # Flux
    flux_ind = flux_df[flux_df['name']==f_name].index.values.astype(int)[0]
    epoch_len = np.floor(flux_df.loc[flux_ind, 'duration']*eeg_fs_down)
    flux_x_in_temp = flux_df.loc[flux_ind, 'flux_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Onset
    onset_ind = onset_df[onset_df['name']==f_name].index.values.astype(int)[0]
    # epoch_len = np.floor(onset_df.loc[onset_ind, 'duration']*eeg_fs_down)
    onset_x_in_temp = onset_df.loc[onset_ind, 'onset_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation LTM
    ltm_exp_ind = ltm_exp_df[ltm_exp_df['name']==f_name].index.values.astype(int)[0]
    ltm_sp_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_hp_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_so_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    ltm_ho_x_in_temp = ltm_exp_df.loc[ltm_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation STM
    stm_exp_ind = stm_exp_df[stm_exp_df['name']==f_name].index.values.astype(int)[0]
    stm_sp_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_hp_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_so_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    stm_ho_x_in_temp = stm_exp_df.loc[stm_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    # Expectation both
    both_exp_ind = both_exp_df[both_exp_df['name']==f_name].index.values.astype(int)[0]
    both_sp_x_in_temp = both_exp_df.loc[both_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_hp_x_in_temp = both_exp_df.loc[both_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_so_x_in_temp = both_exp_df.loc[both_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
    both_ho_x_in_temp = both_exp_df.loc[both_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        
    x_in_temp_A = np.array([flux_x_in_temp,onset_x_in_temp])
    x_in_temp_LTM = np.array([flux_x_in_temp,onset_x_in_temp,ltm_sp_x_in_temp,
                          ltm_hp_x_in_temp,ltm_so_x_in_temp,ltm_ho_x_in_temp])
    x_in_temp_STM = np.array([flux_x_in_temp,onset_x_in_temp,stm_sp_x_in_temp,
                          stm_hp_x_in_temp,stm_so_x_in_temp,stm_ho_x_in_temp])
    x_in_temp_both = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,
                          both_hp_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
    x_in_temp_AM = np.array([flux_x_in_temp,onset_x_in_temp,ltm_sp_x_in_temp,
                          ltm_hp_x_in_temp,ltm_so_x_in_temp,ltm_ho_x_in_temp,
                          stm_sp_x_in_temp,stm_hp_x_in_temp,stm_so_x_in_temp,stm_ho_x_in_temp])
    
    # Transpose to match mTRF toolbox dimension
    x_in_chimera_trim_A += [x_in_temp_A.T]
    x_in_chimera_trim_LTM += [x_in_temp_LTM.T]
    x_in_chimera_trim_STM += [x_in_temp_STM.T]
    x_in_chimera_trim_both += [x_in_temp_both.T]
    x_in_chimera_trim_AM += [x_in_temp_AM.T]
    
# Concatanate both 
x_in_trim_A = x_in_ori_trim_A + x_in_chimera_trim_A
x_in_trim_LTM = x_in_ori_trim_LTM + x_in_chimera_trim_LTM
x_in_trim_STM = x_in_ori_trim_STM + x_in_chimera_trim_STM
x_in_trim_both = x_in_ori_trim_both + x_in_chimera_trim_both
x_in_trim_AM = x_in_ori_trim_AM + x_in_chimera_trim_AM

# %% Get matrix for y
x_out_all = []
x_out_all_sub = []

for subject in subject_list:
    print(subject)
    onset_cut = 1
    exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA/Chimera/'
    regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    
    # Load EEG data
    eeg_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
    epoch_ori = eeg_data['epoch_ori']
    ori_epochs_df = eeg_data['ori_epochs_df']
    epoch_chimera = eeg_data['epoch_chimera']
    chimera_epochs_df = eeg_data['chimera_epochs_df']
    
    x_out = []
    
    for ei in range(n_epoch_ori):
        f_name = all_ori_epochs_df.loc[ei, 'name']
        flux_ind = flux_df[flux_df['name']==f_name].index.values.astype(int)[0]
        epoch_len = np.floor(flux_df.loc[flux_ind, 'duration']*eeg_fs_down)
        eeg_ind = ori_epochs_df[ori_epochs_df['name']==f_name].index.values.astype(int)[0]
        x_out_temp = epoch_ori[eeg_ind,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        x_out += [x_out_temp.T]
        x_out_all += [x_out_temp.T]    

    for ei in range(n_epoch_chimera):
        f_name = all_chimera_epochs_df.loc[ei, 'name']
        flux_ind = flux_df[flux_df['name']==f_name].index.values.astype(int)[0]
        epoch_len = np.floor(flux_df.loc[flux_ind, 'duration']*eeg_fs_down)
        eeg_ind = chimera_epochs_df[chimera_epochs_df['name']==f_name].index.values.astype(int)[0]
        x_out_temp = epoch_chimera[eeg_ind,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        x_out += [x_out_temp.T]
        x_out_all += [x_out_temp.T]
        
    x_out_all_sub += [x_out]
    
#%% loop against CV
from sklearn.model_selection import KFold
reg = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
score = np.zeros((10,7))

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for k, (train_index, test_index) in enumerate(kf.split(np.arange(132))):
    print(k)
    X_train = [x_in_trim_STM[tri] for tri in train_index]
    X_train = X_train*26
    X_test = [x_in_trim_STM[tei] for tei in test_index]
    X_test = X_test*26
    
    y_train = []
    y_test = []
    for i in range(26):
        y_train += [x_out_all_sub[i][tri] for tri in train_index]
        y_test += [x_out_all_sub[i][tei] for tei in test_index]
    for ri, r in enumerate(reg):
        print(r)
        trf_AM = TRF(direction=1)
        trf_AM.train(X_train, y_train, eeg_fs_down, -0.1, 0.7, regularization=r)
        _, metric = trf_AM.predict(X_test, y_test)
        score[k,ri] = metric

np.argmax(np.average(score, axis=0))

# AM=1, LTM=0.1, STM=1, both=1,