#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 17:56:00 2024

@author: tshan
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from mne.filter import resample
from expyfun.io import read_wav, write_hdf5, read_hdf5
from mtrf.model import TRF, load_sample_data
from mtrf.stats import crossval,nested_crossval
from statsmodels.stats import multitest
import mne
#import cupy
import pickle
import matplotlib.pyplot as plt
import os
import time
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
    return y

# %%
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
exp_path = 'F://Chimera/'

# %% Load regressors

do_zscore = True
# Load regressors
regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
regressor_path = 'D://AMPLab/MusicExp/folk_corpus/regressor/125hz/'

# env_df = pd.read_pickle(regressor_path+'env.pkl')
flux_df = pd.read_pickle(regressor_path+'flux.pkl')
onset_df = pd.read_pickle(regressor_path+'onset.pkl')

# expectation LTM
ltm_exp_df = pd.read_pickle(regressor_path+'expectation_ds_ltm_nb_pitch_ioi.pkl')
# expectation STM
stm_exp_df = pd.read_pickle(regressor_path+'expectation_ds_stm_nb_pitch_ioi.pkl')
# expectation both
both_exp_df = pd.read_pickle(regressor_path+'expectation_ds_both_nb_pitch_ioi.pkl')
# Z-score
# if do_zscore:
#     exp_df['sp_ds']
#     exp_df = stats.zscore(, axis=1)

# Normalize regressors
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

# %% TRF

subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

for subject in subject_list:
    
    ti=time.time()
    
    print(subject)
    onset_cut = 1
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    
    # Load set pair value data
    # set_dic = read_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')
    set_dic = read_hdf5('D://AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')

    # Load EEG data
    eeg_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
    epoch_ori = eeg_data['epoch_ori']
    ori_epochs_df = eeg_data['ori_epochs_df']
    epoch_chimera = eeg_data['epoch_chimera']
    chimera_epochs_df = eeg_data['chimera_epochs_df']
    
    ##### Analysis Original
    print("Original")
    # Trim epochs and concatenate epochs
    x_in_ori_trim_both = []
    x_in_ori_trim_both_pitch = []
    x_in_ori_trim_both_time = []
    x_in_ori_trim_A =[]
    
    x_out_ori_trim = []
    f_name_ori = []
    set_num_ori = []
    for ei in range(n_epoch_ori):
        f_name = ori_epochs_df.loc[ei, 'name']
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

        x_in_temp_both = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,
                              both_hp_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
        
        x_in_temp_both_pitch = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,both_hp_x_in_temp])
        
        x_in_temp_both_time = np.array([flux_x_in_temp,onset_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
        

        # Transpose to match mTRF toolbox dimension
        x_in_ori_trim_A += [x_in_temp_A.T]
        x_in_ori_trim_both += [x_in_temp_both.T]
        x_in_ori_trim_both_pitch += [x_in_temp_both_pitch.T]
        x_in_ori_trim_both_time += [x_in_temp_both_time.T]
        # Get x_out
        x_out_temp = epoch_ori[ei,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        #x_out_temp = np.delete(x_out_temp,17, axis=0)
        x_out_ori_trim += [x_out_temp.T] # Transpose to match mTRF toolbox dimension
    
    if subject == 'chimera_012':
        del x_in_ori_trim_A[45:47], x_out_ori_trim[45:47],x_in_ori_trim_both[45:47],x_in_ori_trim_both_pitch[45:47], x_in_ori_trim_both_time[45:47]
    
    ##### Analysis Chimera
    print("Chimera")
    # Trim epochs and concatenate epochs
    x_in_chimera_trim_both = []
    x_in_chimera_trim_both_pitch = []
    x_in_chimera_trim_both_time = []
    x_in_chimera_trim_A = []
    x_out_chimera_trim = []
    f_name_chimera = []
    set_num_chimera = []
    for ei in range(n_epoch_chimera):
        f_name = chimera_epochs_df.loc[ei, 'name']
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
        # Expectation both
        both_exp_ind = both_exp_df[both_exp_df['name']==f_name].index.values.astype(int)[0]
        both_sp_x_in_temp = both_exp_df.loc[both_exp_ind, 'sp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        both_hp_x_in_temp = both_exp_df.loc[both_exp_ind, 'hp_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        both_so_x_in_temp = both_exp_df.loc[both_exp_ind, 'so_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        both_ho_x_in_temp = both_exp_df.loc[both_exp_ind, 'ho_ds_z'][int(onset_cut*eeg_fs_down):int(epoch_len)]
            
        x_in_temp_A = np.array([flux_x_in_temp,onset_x_in_temp])

        x_in_temp_both = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,
                              both_hp_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
        
        x_in_temp_both_pitch = np.array([flux_x_in_temp,onset_x_in_temp,both_sp_x_in_temp,both_hp_x_in_temp])
        
        x_in_temp_both_time = np.array([flux_x_in_temp,onset_x_in_temp,both_so_x_in_temp,both_ho_x_in_temp])
        
        
        # Transpose to match mTRF toolbox dimension
        x_in_chimera_trim_A += [x_in_temp_A.T]
        x_in_chimera_trim_both += [x_in_temp_both.T]
        x_in_chimera_trim_both_pitch += [x_in_temp_both_pitch.T]
        x_in_chimera_trim_both_time += [x_in_temp_both_time.T]
        # Get x_out
        x_out_temp = epoch_chimera[ei,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        x_out_chimera_trim += [x_out_temp.T] # Transpose to match mTRF toolbox dimension

    
    ####### Find best lambda for each model
    print("Find best lambda for each model")
    reg = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    # both
    trf_both = TRF(direction=1)
    trf_both.train(x_in_ori_trim_both+x_in_chimera_trim_both, x_out_ori_trim+x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=reg,k=10)
    both_reg = trf_both.regularization
    
    # both pitch
    trf_both_pitch = TRF(direction=1)
    trf_both_pitch.train(x_in_ori_trim_both_pitch+x_in_chimera_trim_both_pitch, x_out_ori_trim+x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=reg,k=10)
    both_pitch_reg = trf_both_pitch.regularization
    
    # both time
    trf_both_time = TRF(direction=1)
    trf_both_time.train(x_in_ori_trim_both_time+x_in_chimera_trim_both_time, x_out_ori_trim+x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=reg,k=10)
    both_time_reg = trf_both_time.regularization

    # A
    trf_A = TRF(direction=1)
    trf_A.train(x_in_ori_trim_A+x_in_chimera_trim_A, x_out_ori_trim+x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=reg,k=10)
    A_reg = trf_A.regularization

    # Do TRF for each category
    print("fit model...")
    # original
    print("original")
    # both
    trf_ori_both = TRF(direction=1)
    ori_both_correlation, _ = nested_crossval(trf_ori_both, x_in_ori_trim_both, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=both_reg, k=len(x_out_ori_trim))
    
    # both pitch
    trf_ori_both_pitch = TRF(direction=1)
    ori_both_pitch_correlation, _ = nested_crossval(trf_ori_both_pitch, x_in_ori_trim_both_pitch, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=both_pitch_reg, k=len(x_out_ori_trim))
    
    # both time
    trf_ori_both_time = TRF(direction=1)
    ori_both_time_correlation, _ = nested_crossval(trf_ori_both_time, x_in_ori_trim_both_time, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=both_time_reg, k=len(x_out_ori_trim))
    
    # A-only
    # TRF training
    trf_ori_A = TRF(direction=1)
    ori_A_correlation, _ = nested_crossval(trf_ori_A, x_in_ori_trim_A, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=A_reg, k=len(x_out_ori_trim))
    
    
    # Chimera
    print("chimeric")

    # both
    trf_chimera_both = TRF(direction=1)
    chimera_both_correlation, _ = nested_crossval(trf_chimera_both, x_in_chimera_trim_both, x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=both_reg, k=len(x_out_chimera_trim))
    
    # both pitch
    trf_chimera_both_pitch = TRF(direction=1)
    chimera_both_pitch_correlation, _ = nested_crossval(trf_chimera_both_pitch, x_in_chimera_trim_both_pitch, x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=both_pitch_reg, k=len(x_out_chimera_trim))
    
    # both time
    trf_chimera_both_time = TRF(direction=1)
    chimera_both_time_correlation, _ = nested_crossval(trf_chimera_both_time, x_in_chimera_trim_both_time, x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=both_time_reg, k=len(x_out_chimera_trim))
    
    # A-only
    # TRF training
    trf_chimera_A = TRF(direction=1)
    chimera_A_correlation, _ = nested_crossval(trf_chimera_A, x_in_chimera_trim_A, x_out_chimera_trim, eeg_fs_down,-0.1, 0.7, regularization=A_reg, k=len(x_out_chimera_trim))

    # Saving
    write_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_ltm_stm_both_nb_pitch_ioi_hp1_lp8_ICA_trf_data_-02_1_corr_all_crossval_best_reg_individual_pitch-time.hdf5',
               dict(both_reg=both_reg,
                    both_pitch_reg=both_pitch_reg,
                    both_time_reg=both_time_reg,
                    A_reg=A_reg,
                    ori_both_weights=trf_ori_both.weights,
                    ori_bot_pitch_weights=trf_ori_both_pitch.weights,
                    ori_both_time_weights=trf_ori_both_time.weights,
                    ori_A_weights=trf_ori_A.weights,
                    chimera_both_weights=trf_chimera_both.weights,
                    chimera_both_pitch_weights=trf_chimera_both_pitch.weights,
                    chimera_both_time_weights=trf_chimera_both_time.weights,
                    chimera_A_weights=trf_chimera_A.weights,
                    time=trf_ori_A.times,
                    ori_both_correlation=ori_both_correlation,
                    ori_both_pitch_correlation=ori_both_pitch_correlation,
                    ori_both_time_correlation=ori_both_time_correlation,
                    ori_A_correlation=ori_A_correlation,
                    chimera_both_correlation=chimera_both_correlation,
                    chimera_both_pitch_correlation=chimera_both_pitch_correlation,
                    chimera_both_time_correlation=chimera_both_time_correlation,
                    chimera_A_correlation=chimera_A_correlation,
                    set_num_ori=set_num_ori,
                    set_num_chimera=set_num_chimera),
               overwrite=True)
    
    print(time.time()-ti)