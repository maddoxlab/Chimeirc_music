#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 14:27:38 2023

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from mne.filter import resample
from expyfun.io import read_wav, write_hdf5, read_hdf5
from mtrf.model import TRF, load_sample_data
from mtrf.stats import cross_validate
from statsmodels.stats import multitest
import mne
#import cupy
import pickle
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
    return y

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
regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
# %% Derive ERP
# ERP params
erp_start = -0.05
erp_stop = 0.7
step = int(1/eeg_fs_down*1e3)
lag = np.arange(erp_start*1e3,erp_stop*1e3+step, step=step)
# %% Get EEG data, save into one file
import warnings
from tqdm import tqdm
import time
warnings.filterwarnings('ignore')

subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']
subject_num = len(subject_list)

df = pd.DataFrame(columns=['category','group','fname','note_num','sp','hp','so','ho','channel','erp'])
df['erp'] = df['erp'].astype(object)

# Create an outer progress bar
outer_progress_bar = tqdm(total=len(subject_list), desc='Outer Loop')

for subject,i in zip(subject_list, range(len(subject_list))):
    outer_progress_bar.update(1)
    # Define dataframe
    df = pd.DataFrame(columns=['category','group','fname','note_num','sp','hp','so','ho','channel','erp'])
    df['erp'] = df['erp'].astype(object)
    
    print(subject)
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    
    # Load set pair value data
    set_dic = read_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')
    # Load regressors
    # env_df = pd.read_pickle(regressor_path+'env.pkl')
    flux_df = pd.read_pickle(regressor_path+'flux.pkl')
    onset_df = pd.read_pickle(regressor_path+'onset.pkl')
    
    note_dure = []
    for ei in range(n_epoch_ori):
        onset_index = [np.where(onset_df['onset_ds'][ei]==1)]
        for i in range(len(onset_index)-1):
            note_dure += []
    # expectation
    exp_df = pd.read_pickle(regressor_path+'expectation_downbeat_seq_ltm_b16.pkl')
    
    # Load EEG data
    eeg_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
    epoch_ori = eeg_data['epoch_ori']
    ori_epochs_df = eeg_data['ori_epochs_df']
    epoch_chimera = eeg_data['epoch_chimera']
    chimera_epochs_df = eeg_data['chimera_epochs_df']
    
    # re-epoching Original
    # Create an inner progress bar
    inner_progress_bar = tqdm(total=n_epoch_ori, desc='Inner Loop', leave=False, position=1)
    info = mne.create_info(channel_names, sfreq=eeg_fs_down,ch_types='eeg')
    for ei in range(n_epoch_ori):
        # Update the inner progress bar
        inner_progress_bar.update(1)
        print(subject+' original'+str(ei))
        f_name = ori_epochs_df.loc[ei, 'name']
        f_name_ori = f_name[:-4]
        set_num_ori = set_dic[f_name_ori]
        onset_list_ind = onset_df[onset_df['name']==f_name].index.values.astype(int)[0]
        onset_x_in_temp = onset_df.loc[onset_list_ind, 'onset_ds']
        event_col_index = np.where(onset_x_in_temp==1)[0]
        event_col_id = np.ones_like(event_col_index)
        event_col_id[0],event_col_id[-1] = 0, 0 # don't count in the first and the last note
        events = np.column_stack((event_col_index,np.zeros_like(event_col_index),event_col_id))
        raw_trial = mne.io.RawArray(epoch_ori[ei], info)
        note_epochs = mne.Epochs(raw_trial, events, event_id=1, tmin=erp_start, tmax=erp_stop, baseline=None, preload=True, verbose="WARNING")
        note_epochs_data = note_epochs.get_data()
        
        
        exp_ind = exp_df[exp_df['name']==f_name].index.values.astype(int)[0]
        sp_x_in_temp = exp_df.loc[exp_ind, 'sp']
        sp_x_in_temp = sp_x_in_temp[1:-1] # remove first and last notes
        hp_x_in_temp = exp_df.loc[exp_ind, 'hp']
        hp_x_in_temp = hp_x_in_temp[1:-1]
        so_x_in_temp = exp_df.loc[exp_ind, 'so']
        so_x_in_temp = so_x_in_temp[1:-1]
        ho_x_in_temp = exp_df.loc[exp_ind, 'ho']
        ho_x_in_temp = ho_x_in_temp[1:-1]
        downbeat_pitch_temp = exp_df.loc[exp_ind, 'downbeat_pitch']
        downbeat_pitch_temp = downbeat_pitch_temp[1:-1]
        downbeat_onset_temp = exp_df.loc[exp_ind, 'downbeat_onset']
        downbeat_onset_temp = downbeat_onset_temp[1:-1]
        for ni in range(note_epochs_data.shape[0]):
            for chn in range(len(channel_names)):
                df = df.append({'subject':subject,'category':'original','group':set_num_ori,
                                'fname': f_name_ori,'note_num':ni+1,
                                'sp':sp_x_in_temp[ni],'hp':hp_x_in_temp[ni],
                                'so':so_x_in_temp[ni],'ho':ho_x_in_temp[ni],
                                'downbeat_pitch':downbeat_pitch_temp[ni],
                                'downbeat_onset': downbeat_onset_temp[ni],
                                'channel': channel_names[chn],
                                'erp':[note_epochs_data[ni,chn,:]]},ignore_index=True)
        # Close the inner progress bar
        inner_progress_bar.close()
    
    # re-epoching Chimera
    # Create an inner progress bar
    inner_progress_bar = tqdm(total=n_epoch_chimera, desc='Inner Loop', leave=False, position=1)
    
    info = mne.create_info(channel_names, sfreq=eeg_fs_down,ch_types='eeg')
    for ei in range(n_epoch_chimera):
        inner_progress_bar.update(1)
        print(subject+' chimera'+str(ei))
        f_name = chimera_epochs_df.loc[ei, 'name']
        f_name_chimera = f_name[:-4]
        set_num_chimera = set_dic[f_name_chimera]
        onset_list_ind = onset_df[onset_df['name']==f_name].index.values.astype(int)[0]
        onset_x_in_temp = onset_df.loc[onset_list_ind, 'onset_ds']
        event_col_index = np.where(onset_x_in_temp==1)[0]
        event_col_id = np.ones_like(event_col_index)
        event_col_id[0],event_col_id[-1] = 0, 0 # don't count in the first and the last note
        events = np.column_stack((event_col_index,np.zeros_like(event_col_index),event_col_id))
        raw_trial = mne.io.RawArray(epoch_chimera[ei], info)
        note_epochs = mne.Epochs(raw_trial, events, event_id=1, tmin=erp_start, tmax=erp_stop, baseline=None, preload=True,verbose="WARNING")
        note_epochs_data = note_epochs.get_data()
        
        exp_ind = exp_df[exp_df['name']==f_name].index.values.astype(int)[0]
        sp_x_in_temp = exp_df.loc[exp_ind, 'sp']
        sp_x_in_temp = sp_x_in_temp[1:-1] # remove first and last notes
        hp_x_in_temp = exp_df.loc[exp_ind, 'hp']
        hp_x_in_temp = hp_x_in_temp[1:-1]
        so_x_in_temp = exp_df.loc[exp_ind, 'so']
        so_x_in_temp = so_x_in_temp[1:-1]
        ho_x_in_temp = exp_df.loc[exp_ind, 'ho']
        ho_x_in_temp = ho_x_in_temp[1:-1]
        downbeat_pitch_temp = exp_df.loc[exp_ind, 'downbeat_pitch']
        downbeat_pitch_temp = downbeat_pitch_temp[1:-1]
        downbeat_onset_temp = exp_df.loc[exp_ind, 'downbeat_onset']
        downbeat_onset_temp = downbeat_onset_temp[1:-1]
        for ni in range(note_epochs_data.shape[0]):
            for chn in range(len(channel_names)):
                df = df.append({'subject':subject,'category':'chimera','group':set_num_chimera,
                                'fname': f_name_chimera,'note_num':ni+1,
                                'sp':sp_x_in_temp[ni],'hp':hp_x_in_temp[ni],
                                'so':so_x_in_temp[ni],'ho':ho_x_in_temp[ni],
                                'downbeat_pitch':downbeat_pitch_temp[ni],
                                'downbeat_onset': downbeat_onset_temp[ni],
                                'channel': channel_names[chn],
                                'erp':[note_epochs_data[ni,chn,:]]},ignore_index=True)
        # Close the inner progress bar
        inner_progress_bar.close()
                
    df.to_pickle(exp_path+'/'+subject+"_note_ERP_expectation_downbeat.pkl")
    # Close the outer progress bar
    outer_progress_bar.close()


# %% All ERPS statistics
# %% All ERPS statistics
ERP_all_df = pd.DataFrame(columns=['subject','category','channel',
                                   'downbeat_pitch_onset','downbeat_pitch_n',
                                   'downbeat_n_onset','downbeat_n_n'])
for channel in channel_names:
    print(channel)
    for subject,i in zip(subject_list, range(len(subject_list))):
        print(subject)
        df = pd.read_pickle(exp_path+'/'+subject+"_note_ERP_expectation_downbeat.pkl")
        downbeat_pitch_onset_original_set = df[(df['downbeat_pitch'] == 1) & (df['downbeat_onset'] == 1) & (df['channel'] == channel) & (df['category'] == 'original')]
        # downbeat_pitch_n_original_set = df[(df['downbeat_pitch'] == 1) & (df['downbeat_onset'] == 0) & (df['channel'] == channel) & (df['category'] == 'original')]
        # downbeat_n_onset_original_set = df[(df['downbeat_pitch'] == 0) & (df['downbeat_onset'] == 1) & (df['channel'] == channel) & (df['category'] == 'original')]
        downbeat_n_n_original_set = df[(df['downbeat_pitch'] == 0) & (df['downbeat_onset'] == 0) & (df['channel'] == channel) & (df['category'] == 'original')]

        downbeat_pitch_onset_chimera_set = df[(df['downbeat_pitch'] == 1) & (df['downbeat_onset'] == 1) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        downbeat_pitch_n_chimera_set = df[(df['downbeat_pitch'] == 1) & (df['downbeat_onset'] == 0) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        downbeat_n_onset_chimera_set = df[(df['downbeat_pitch'] == 0) & (df['downbeat_onset'] == 1) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        downbeat_n_n_chimera_set = df[(df['downbeat_pitch'] == 0) & (df['downbeat_onset'] == 0) & (df['channel'] == channel) & (df['category'] == 'chimera')]


        downbeat_pitch_onset_ori_erp = np.squeeze(np.array(downbeat_pitch_onset_original_set['erp'].to_list()), axis=1)
        downbeat_pitch_onset_ori_erp_ave = np.average(downbeat_pitch_onset_ori_erp, axis=0)
        # downbeat_pitch_n_ori_erp = np.squeeze(np.array(downbeat_pitch_n_original_set['erp'].to_list()), axis=1)
        # downbeat_pitch_n_ori_erp_ave = np.average(downbeat_pitch_n_ori_erp, axis=0)
        # downbeat_n_onset_ori_erp = np.squeeze(np.array(downbeat_n_onset_original_set['erp'].to_list()), axis=1)
        # downbeat_n_onset_ori_erp_ave = np.average(downbeat_n_onset_ori_erp, axis=0)
        downbeat_n_n_ori_erp = np.squeeze(np.array(downbeat_n_n_original_set['erp'].to_list()), axis=1)
        downbeat_n_n_ori_erp_ave = np.average(downbeat_n_n_ori_erp, axis=0)

        ERP_all_df = ERP_all_df.append({'subject':subject,'category':'original','channel':channel,
                           'downbeat_pitch_onset':downbeat_pitch_onset_ori_erp_ave,
                           # 'downbeat_pitch_n':downbeat_pitch_n_ori_erp_ave,
                           # 'downbeat_n_onset':downbeat_n_onset_ori_erp_ave,
                           'downbeat_n_n':downbeat_n_n_ori_erp_ave},ignore_index=True)
        
        downbeat_pitch_onset_chimera_erp = np.squeeze(np.array(downbeat_pitch_onset_chimera_set['erp'].to_list()), axis=1)
        downbeat_pitch_onset_chimera_erp_ave = np.average(downbeat_pitch_onset_chimera_erp, axis=0)
        downbeat_pitch_n_chimera_erp = np.squeeze(np.array(downbeat_pitch_n_chimera_set['erp'].to_list()), axis=1)
        downbeat_pitch_n_chimera_erp_ave = np.average(downbeat_pitch_n_chimera_erp, axis=0)
        downbeat_n_onset_chimera_erp = np.squeeze(np.array(downbeat_n_onset_chimera_set['erp'].to_list()), axis=1)
        downbeat_n_onset_chimera_erp_ave = np.average(downbeat_n_onset_chimera_erp, axis=0)
        downbeat_n_n_chimera_erp = np.squeeze(np.array(downbeat_n_n_chimera_set['erp'].to_list()), axis=1)
        downbeat_n_n_chimera_erp_ave = np.average(downbeat_n_n_chimera_erp, axis=0)

        ERP_all_df = ERP_all_df.append({'subject':subject,'category':'chimera','channel':channel,
                           'downbeat_pitch_onset':downbeat_pitch_onset_chimera_erp_ave,
                           'downbeat_pitch_n':downbeat_pitch_n_chimera_erp_ave,
                           'downbeat_n_onset':downbeat_n_onset_chimera_erp_ave,
                           'downbeat_n_n':downbeat_n_n_chimera_erp_ave},ignore_index=True)
        
ERP_all_df.to_pickle(exp_path+'/note_ERP_downbeat_all.pkl')

# %%
# 
# Original downbeat effect
channel = 'Fz'
ERP_original = ERP_all_df[(ERP_all_df['category']=='original') & (ERP_all_df['channel']==channel)]

ERP_original_downbeat = np.array(ERP_original['downbeat_pitch_onset'].to_list())
ERP_original_downbeat_ave = np.average(ERP_original_downbeat, axis=0)
ERP_original_downbeat_err = np.std(ERP_original_downbeat, axis=0)/np.sqrt(subject_num)

ERP_original_n = np.array(ERP_original['downbeat_n_n'].to_list())
ERP_original_n_ave = np.average(ERP_original_n, axis=0)
ERP_original_n_err = np.std(ERP_original_n, axis=0)/np.sqrt(subject_num)

original_diff = ERP_original_downbeat - ERP_original_n
original_diff_ave = np.average(original_diff, axis=0)
original_diff_err = np.std(original_diff_ave, axis=0)/np.sqrt(subject_num)

# Chimera downbeat effect
ERP_chimera = ERP_all_df[(ERP_all_df['category']=='chimera') & (ERP_all_df['channel']==channel)]

ERP_chimera_downbeat = np.array(ERP_chimera['downbeat_pitch_onset'].to_list())
ERP_chimera_downbeat_ave = np.average(ERP_chimera_downbeat, axis=0)
ERP_chimera_downbeat_err = np.std(ERP_chimera_downbeat, axis=0)/np.sqrt(subject_num)

ERP_chimera_n_n = np.array(ERP_chimera['downbeat_n_n'].to_list())
ERP_chimera_n_n_ave = np.average(ERP_chimera_n_n, axis=0)
ERP_chimera_n_n_err = np.std(ERP_original_n, axis=0)/np.sqrt(subject_num)

ERP_chimera_pitch_n = np.array(ERP_chimera['downbeat_pitch_n'].to_list())
ERP_chimera_pitch_n_ave = np.average(ERP_chimera_pitch_n, axis=0)
ERP_chimera_pitch_n_err = np.std(ERP_chimera_pitch_n, axis=0)/np.sqrt(subject_num)

ERP_chimera_n_onset = np.array(ERP_chimera['downbeat_n_onset'].to_list())
ERP_chimera_n_onset_ave = np.average(ERP_chimera_n_onset, axis=0)
ERP_chimera_n_onset_err = np.std(ERP_chimera_n_onset, axis=0)/np.sqrt(subject_num)

chimera_diff_down_n = ERP_chimera_downbeat - ERP_chimera_n_n
chimera_diff_down_n_ave = np.average(chimera_diff_down_n, axis=0)
chimera_diff_down_n_err = np.std(chimera_diff_down_n, axis=0)/np.sqrt(subject_num)

chimera_diff_down_pitch = ERP_chimera_pitch_n - ERP_chimera_n_n
chimera_diff_down_pitch_ave = np.average(chimera_diff_down_pitch, axis=0)
chimera_diff_down_pitch_err = np.std(chimera_diff_down_pitch, axis=0)/np.sqrt(subject_num)

chimera_diff_down_onset = ERP_chimera_n_onset - ERP_chimera_n_n
chimera_diff_down_onset_ave = np.average(chimera_diff_down_onset, axis=0)
chimera_diff_down_onset_err = np.std(chimera_diff_down_onset, axis=0)/np.sqrt(subject_num)


plt.figure()
plt.plot(lag, ERP_original_downbeat_ave, c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, ERP_original_downbeat_ave-ERP_original_downbeat_err, ERP_original_downbeat_ave+ERP_original_downbeat_err, color='C0', alpha=0.8)
plt.plot(lag, ERP_original_n_ave, c='C1', linewidth=2, linestyle='solid', label="Original-other")
plt.fill_between(lag, ERP_original_n_ave-ERP_original_n_err, ERP_original_n_ave+ERP_original_n_err, color='C1',alpha=0.5)
plt.plot(lag, ERP_chimera_downbeat_ave, c='C0', linewidth=1, linestyle='dashed', label="Chimera-Downbeat")
plt.fill_between(lag, ERP_chimera_downbeat_ave-ERP_chimera_downbeat_err, ERP_chimera_downbeat_ave+ERP_chimera_downbeat_err, color='C0', alpha=0.8)
plt.plot(lag, ERP_chimera_n_n_ave, c='C1', linewidth=1, linestyle='dashed', label="Chimera-other")
plt.fill_between(lag, ERP_chimera_n_n_ave-ERP_chimera_n_n_err, ERP_chimera_n_n_ave+ERP_chimera_n_n_err, color='C1',alpha=0.5)
plt.plot(lag, ERP_chimera_pitch_n_ave, c='C2', linewidth=1, linestyle='dashed', label="Chimera-pitch")
plt.fill_between(lag, ERP_chimera_pitch_n_ave-ERP_chimera_pitch_n_err, ERP_chimera_pitch_n_ave+ERP_chimera_pitch_n_err, color='C2',alpha=0.5)
plt.plot(lag, ERP_chimera_n_onset_ave, c='C3', linewidth=1, linestyle='dashed', label="Chimera-onset")
plt.fill_between(lag, ERP_chimera_n_onset_ave-ERP_chimera_n_onset_err, ERP_chimera_n_onset_ave+ERP_chimera_n_onset_err, color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title(channel)
plt.legend()
plt.show()

# Difference
plt.figure()
plt.plot(lag, original_diff_ave, c='C0', linewidth=2, linestyle='solid', label="Original: Downbeat vs other")
plt.fill_between(lag, original_diff_ave-original_diff_err, original_diff_ave+original_diff_err, color='C0', alpha=0.8)
plt.plot(lag, chimera_diff_down_n_ave, c='C1', linewidth=2, linestyle='solid', label="Chimera: Downbeat vs other")
plt.fill_between(lag, chimera_diff_down_n_ave-chimera_diff_down_n_err, chimera_diff_down_n_ave+chimera_diff_down_n_err, color='C1', alpha=0.8)
plt.plot(lag, chimera_diff_down_pitch_ave, c='C2', linewidth=2, linestyle='solid', label="Chimera: pitch vs other")
plt.fill_between(lag, chimera_diff_down_pitch_ave-chimera_diff_down_pitch_err, chimera_diff_down_pitch_ave+chimera_diff_down_pitch_err, color='C2', alpha=0.8)
plt.plot(lag, chimera_diff_down_onset_ave, c='C3', linewidth=2, linestyle='solid', label="Chimera: onset vs other")
plt.fill_between(lag, chimera_diff_down_onset_ave-chimera_diff_down_onset_err, chimera_diff_down_onset_ave+chimera_diff_down_onset_err, color='C3', alpha=0.8)
plt.grid()
plt.title(channel)
plt.legend()
plt.show()

# %% Topographs
ERP_original_downbeat_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_original_n_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_downbeat_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_n_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_pitch_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_onset_all = np.zeros((subject_num, len(channel_names), len(lag)))

ERP_original_downbeat_n_diff_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_downbeat_n_diff_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_pitch_n_diff_all = np.zeros((subject_num, len(channel_names), len(lag)))
ERP_chimera_onset_n_diff_all = np.zeros((subject_num, len(channel_names), len(lag)))


for chi, channel in enumerate(channel_names):
    ERP_original = ERP_all_df[(ERP_all_df['category']=='original') & (ERP_all_df['channel']==channel)]
    ERP_original_downbeat_all[:,chi,:] = np.array(ERP_original['downbeat_pitch_onset'].to_list())
    ERP_original_n_all[:,chi,:] = np.array(ERP_original['downbeat_n_n'].to_list())
    ERP_original_downbeat_n_diff_all[:,chi,:] = ERP_original_downbeat_all[:,chi,:] - ERP_original_n_all[:,chi,:]
    
    ERP_chimera = ERP_all_df[(ERP_all_df['category']=='chimera') & (ERP_all_df['channel']==channel)]
    ERP_chimera_downbeat_all[:,chi,:] = np.array(ERP_chimera['downbeat_pitch_onset'].to_list())
    ERP_chimera_n_all[:,chi,:] = np.array(ERP_chimera['downbeat_n_n'].to_list())
    ERP_chimera_pitch_all[:,chi,:] = np.array(ERP_chimera['downbeat_pitch_n'].to_list())
    ERP_chimera_onset_all[:,chi,:] = np.array(ERP_chimera['downbeat_n_onset'].to_list())
    ERP_chimera_downbeat_n_diff_all[:,chi,:] = ERP_chimera_downbeat_all[:,chi,:] - ERP_chimera_n_all[:,chi,:] 
    ERP_chimera_pitch_n_diff_all[:,chi,:] = ERP_chimera_pitch_all[:,chi,:] - ERP_chimera_n_all[:,chi,:]
    ERP_chimera_onset_n_diff_all[:,chi,:] = ERP_chimera_onset_all[:,chi,:] - ERP_chimera_n_all[:,chi,:]
    
# Statistics
# Weight Stats
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
p_raw = np.zeros((4,len(channel_names),95))
sig_time = np.zeros((4,len(channel_names),95))
p_corrected = np.zeros((4,len(channel_names),95))
sig_time_corrected = np.zeros((4,len(channel_names),95))
alpha=0.05
for chi in range(32):
    res = stats.permutation_test((ERP_original_downbeat_all[:,chi,:], ERP_original_n_all[:,chi,:]),statistic,permutation_type='samples',axis=0)
    p_raw[0,chi,:] = res.pvalue
    indices = np.where(p_raw[0,chi,:]<=alpha)[0]
    sig_time[0,chi,indices] = 1
    sig_time_corrected[0,chi,:], p_corrected[0,chi,:] = multitest.fdrcorrection(res.pvalue,alpha=alpha)
    
    res = stats.permutation_test((ERP_chimera_downbeat_all[:,chi,:], ERP_chimera_n_all[:,chi,:]),statistic,permutation_type='samples',axis=0)
    p_raw[1,chi,:] = res.pvalue
    indices = np.where(p_raw[1,chi,:]<=alpha)[0]
    sig_time[1,chi,indices] = 1
    sig_time_corrected[1,chi,:], p_corrected[1,chi,:] = multitest.fdrcorrection(res.pvalue,alpha=alpha)
    
    res = stats.permutation_test((ERP_chimera_pitch_all[:,chi,:], ERP_chimera_n_all[:,chi,:]),statistic,permutation_type='samples',axis=0)
    p_raw[2,chi,:] = res.pvalue
    indices = np.where(p_raw[2,chi,:]<=alpha)[0]
    sig_time[2,chi,indices] = 1
    sig_time_corrected[2,chi,:], p_corrected[2,chi,:] = multitest.fdrcorrection(res.pvalue,alpha=alpha)
    
    res = stats.permutation_test((ERP_chimera_onset_all[:,chi,:], ERP_chimera_n_all[:,chi,:]),statistic,permutation_type='samples',axis=0)
    p_raw[3,chi,:] = res.pvalue
    indices = np.where(p_raw[3,chi,:]<=alpha)[0]
    sig_time[3,chi,indices] = 1
    sig_time_corrected[3,chi,:], p_corrected[3,chi,:] = multitest.fdrcorrection(res.pvalue,alpha=alpha)

    

chi_n=1
plt.figure()
plt.plot(lag, original_diff_ave, c='C0', linewidth=2, linestyle='solid', label="Original: Downbeat vs other")
plt.fill_between(lag, original_diff_ave-original_diff_err, original_diff_ave+original_diff_err, color='C0', alpha=0.8)
plt.fill_between(lag, -3.5e-6, 3.5e-6, where=(sig_time_corrected[0,chi_n,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
plt.ylim(-1e-6, 1.5e-6)
plt.grid()
plt.title(channel_names[chi_n])
plt.legend()
plt.show()

chi_n=1
plt.figure()
plt.plot(lag, chimera_diff_down_n_ave, c='C1', linewidth=2, linestyle='solid', label="Chimera: Downbeat vs other")
plt.fill_between(lag, chimera_diff_down_n_ave-chimera_diff_down_n_err, chimera_diff_down_n_ave+chimera_diff_down_n_err, color='C1', alpha=0.8)
plt.fill_between(lag, -3.5e-6, 3.5e-6, where=(sig_time_corrected[1,chi_n,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
plt.ylim(-1e-6, 1.5e-6)
plt.grid()
plt.title(channel_names[chi_n])
plt.legend()
plt.show()

chi_n=1
plt.figure()
plt.plot(lag, chimera_diff_down_pitch_ave, c='C2', linewidth=2, linestyle='solid', label="Chimera: pitch vs other")
plt.fill_between(lag, chimera_diff_down_pitch_ave-chimera_diff_down_pitch_err, chimera_diff_down_pitch_ave+chimera_diff_down_pitch_err, color='C2', alpha=0.8)
plt.fill_between(lag, -3.5e-6, 3.5e-6, where=(sig_time_corrected[2,chi_n,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
plt.ylim(-1e-6, 1.5e-6)
plt.grid()
plt.title(channel_names[chi_n])
plt.legend()
plt.show()

chi_n=1
plt.figure()
plt.plot(lag, chimera_diff_down_onset_ave, c='C3', linewidth=2, linestyle='solid', label="Chimera: onset vs other")
plt.fill_between(lag, chimera_diff_down_onset_ave-chimera_diff_down_onset_err, chimera_diff_down_onset_ave+chimera_diff_down_onset_err, color='C3', alpha=0.8)
plt.fill_between(lag, -3.5e-6, 3.5e-6, where=(sig_time_corrected[3,chi_n,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
plt.ylim(-1e-6, 1.5e-6)
plt.grid()
plt.title(channel_names[chi_n])
plt.legend()
plt.show()

# Topoplots
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(channel_names, sfreq=eeg_fs_down, ch_types='eeg')
# Original
ERP_original_downbeat_all_ave = np.average(ERP_original_downbeat_all, axis=0)
ERP_original_downbeat_ep = mne.EvokedArray(ERP_original_downbeat_all_ave, info, tmin=-0.05)
ERP_original_downbeat_ep.set_montage(montage)
ERP_original_downbeat_ep.nave = subject_num
ERP_original_downbeat_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - downbeat")

ERP_original_n_all_ave = np.average(ERP_original_n_all, axis=0)
ERP_original_n_ep = mne.EvokedArray(ERP_original_n_all_ave, info, tmin=-0.05)
ERP_original_n_ep.set_montage(montage)
ERP_original_n_ep.nave = subject_num
ERP_original_n_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - other")
# Chimera
ERP_chimera_downbeat_all_ave = np.average(ERP_chimera_downbeat_all, axis=0)
ERP_chimera_downbeat_ep = mne.EvokedArray(ERP_chimera_downbeat_all_ave, info, tmin=-0.05)
ERP_chimera_downbeat_ep.set_montage(montage)
ERP_chimera_downbeat_ep.nave = subject_num
ERP_chimera_downbeat_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - downbeat")

ERP_chimera_n_all_ave = np.average(ERP_chimera_n_all, axis=0)
ERP_chimera_n_ep = mne.EvokedArray(ERP_chimera_n_all_ave, info, tmin=-0.05)
ERP_chimera_n_ep.set_montage(montage)
ERP_chimera_n_ep.nave = subject_num
ERP_chimera_n_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - other")


ERP_chimera_pitch_all_ave = np.average(ERP_chimera_pitch_all, axis=0)
ERP_chimera_pitch_ep = mne.EvokedArray(ERP_chimera_pitch_all_ave, info, tmin=-0.05)
ERP_chimera_pitch_ep.set_montage(montage)
ERP_chimera_pitch_ep.nave = subject_num
ERP_chimera_pitch_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - pitch")

ERP_chimera_onset_all_ave = np.average(ERP_chimera_onset_all, axis=0)
ERP_chimera_onset_ep = mne.EvokedArray(ERP_chimera_onset_all_ave, info, tmin=-0.05)
ERP_chimera_onset_ep.set_montage(montage)
ERP_chimera_onset_ep.nave = subject_num
ERP_chimera_onset_ep.plot_joint(times = [65e-3,185e-3,300e-3,370e-3,430e-3],
                            ts_args=dict(ylim=dict(eeg=[-3.5e-6,3.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-3.5e-6, vmax=3.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - onset")


# Differences
ERP_original_downbeat_n_diff_all_ave = np.average(ERP_original_downbeat_n_diff_all, axis=0)
ERP_original_downbeat_n_diff_ep = mne.EvokedArray(ERP_original_downbeat_n_diff_all_ave, info, tmin=-0.05)
ERP_original_downbeat_n_diff_ep.set_montage(montage)
ERP_original_downbeat_n_diff_ep.nave = subject_num
ERP_original_downbeat_n_diff_ep.plot_joint(times = [150e-3,270e-3,350e-3,420e-3,560e-3],
                            ts_args=dict(ylim=dict(eeg=[-1.5e-6,1.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-1.5e-6, vmax=1.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Downbeat vs other")

ERP_chimera_downbeat_n_diff_all_ave = np.average(ERP_chimera_downbeat_n_diff_all, axis=0)
ERP_chimera_downbeat_n_diff_ep = mne.EvokedArray(ERP_chimera_downbeat_n_diff_all_ave, info, tmin=-0.05)
ERP_chimera_downbeat_n_diff_ep.set_montage(montage)
ERP_chimera_downbeat_n_diff_ep.nave = subject_num
ERP_chimera_downbeat_n_diff_ep.plot_joint(times = [150e-3,270e-3,350e-3,420e-3,560e-3],
                            ts_args=dict(ylim=dict(eeg=[-1.5e-6,1.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-1.5e-6, vmax=1.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Downbeat vs other")

ERP_chimera_pitch_n_diff_all_ave = np.average(ERP_chimera_pitch_n_diff_all, axis=0)
ERP_chimera_pitch_n_diff_ep = mne.EvokedArray(ERP_chimera_pitch_n_diff_all_ave, info, tmin=-0.05)
ERP_chimera_pitch_n_diff_ep.set_montage(montage)
ERP_chimera_pitch_n_diff_ep.nave = subject_num
ERP_chimera_pitch_n_diff_ep.plot_joint(times = [150e-3,270e-3,350e-3,420e-3,560e-3],
                            ts_args=dict(ylim=dict(eeg=[-0.5e-6,0.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-0.5e-6, vmax=0.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Pitch vs other")


ERP_chimera_onset_n_diff_all_ave = np.average(ERP_chimera_onset_n_diff_all, axis=0)
ERP_chimera_onset_n_diff_ep = mne.EvokedArray(ERP_chimera_onset_n_diff_all_ave, info, tmin=-0.05)
ERP_chimera_onset_n_diff_ep.set_montage(montage)
ERP_chimera_onset_n_diff_ep.nave = subject_num
ERP_chimera_onset_n_diff_ep.plot_joint(times = [150e-3,270e-3,350e-3,420e-3,560e-3],
                            ts_args=dict(ylim=dict(eeg=[-1.5e-6,1.5e-6]), xlim=[-50, 700],
                                         time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vmin=-1.5e-6, vmax=1.5e-6, time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Onset vs other")



    