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
    exp_df = pd.read_pickle(regressor_path+'expectation_ds_ltm_b4.pkl')
    
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
        sp_x_in_temp = exp_df.loc[exp_ind, 'sp_ds']
        sp_x_in_temp = sp_x_in_temp[np.nonzero(sp_x_in_temp)][1:-1] # remove first and last notes
        hp_x_in_temp = exp_df.loc[exp_ind, 'hp_ds']
        hp_x_in_temp = hp_x_in_temp[np.nonzero(hp_x_in_temp)][1:-1]
        so_x_in_temp = exp_df.loc[exp_ind, 'so_ds']
        so_x_in_temp = so_x_in_temp[np.nonzero(so_x_in_temp)][1:-1]
        ho_x_in_temp = exp_df.loc[exp_ind, 'ho_ds']
        ho_x_in_temp = ho_x_in_temp[np.nonzero(ho_x_in_temp)][1:-1]
        for ni in range(note_epochs_data.shape[0]):
            for chn in range(len(channel_names)):
                df = df.append({'subject':subject,'category':'original','group':set_num_ori,
                                'fname': f_name_ori,'note_num':ni+1,
                                'sp':sp_x_in_temp[ni],'hp':hp_x_in_temp[ni],
                                'so':so_x_in_temp[ni],'ho':ho_x_in_temp[ni],
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
        sp_x_in_temp = exp_df.loc[exp_ind, 'sp_ds']
        sp_x_in_temp = sp_x_in_temp[np.nonzero(sp_x_in_temp)][1:-1] # remove first and last notes
        hp_x_in_temp = exp_df.loc[exp_ind, 'hp_ds']
        hp_x_in_temp = hp_x_in_temp[np.nonzero(hp_x_in_temp)][1:-1]
        so_x_in_temp = exp_df.loc[exp_ind, 'so_ds']
        so_x_in_temp = so_x_in_temp[np.nonzero(so_x_in_temp)][1:-1]
        ho_x_in_temp = exp_df.loc[exp_ind, 'ho_ds']
        ho_x_in_temp = ho_x_in_temp[np.nonzero(ho_x_in_temp)][1:-1]
        for ni in range(note_epochs_data.shape[0]):
            for chn in range(len(channel_names)):
                df = df.append({'subject':subject,'category':'chimera','group':set_num_chimera,
                                'fname': f_name_chimera,'note_num':ni+1,
                                'sp':sp_x_in_temp[ni],'hp':hp_x_in_temp[ni],
                                'so':so_x_in_temp[ni],'ho':ho_x_in_temp[ni],
                                'channel': channel_names[chn],
                                'erp':[note_epochs_data[ni,chn,:]]},ignore_index=True)
        # Close the inner progress bar
        inner_progress_bar.close()
                
    df.to_pickle(exp_path+'/'+subject+"_note_ERP_expectation.pkl")
    # Close the outer progress bar
    outer_progress_bar.close()

# %% Find the top and bottom 15% pitch, onset, expectation
exp_qt = pd.read_pickle(regressor_path + '/expectation_qt.pkl')
qt = 0.15
high_sp = exp_qt[exp_qt['qts']==qt]['high_sp'].values[0]
high_hp = exp_qt[exp_qt['qts']==qt]['high_hp'].values[0]
high_so = exp_qt[exp_qt['qts']==qt]['high_so'].values[0]
high_ho = exp_qt[exp_qt['qts']==qt]['high_ho'].values[0]
low_sp = exp_qt[exp_qt['qts']==qt]['low_sp'].values[0]
low_hp = exp_qt[exp_qt['qts']==qt]['low_hp'].values[0]
low_so = exp_qt[exp_qt['qts']==qt]['low_so'].values[0]
low_ho = exp_qt[exp_qt['qts']==qt]['low_ho'].values[0]


# %% All ERPS statistics
ERP_all_df = pd.DataFrame(columns=['subject','category','channel',
                                   'high_sp','high_hp','high_so','high_ho',
                                   'low_sp','low_hp','low_so','low_ho'])
for channel in channel_names:
    print(channel)
    for subject,i in zip(subject_list, range(len(subject_list))):
        print(subject)
        df = pd.read_pickle(exp_path+'/'+subject+"_note_ERP_expectation.pkl")
        high_sp_original_set = df[(df['sp'] >= high_sp) & (df['channel'] == channel) & (df['category'] == 'original')]
        high_sp_chimera_set = df[(df['sp'] >= high_sp) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        high_hp_original_set = df[(df['hp'] >= high_hp) & (df['channel'] == channel) & (df['category'] == 'original')]
        high_hp_chimera_set = df[(df['hp'] >= high_hp) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        high_so_original_set = df[(df['so'] >= high_so) & (df['channel'] == channel) & (df['category'] == 'original')]
        high_so_chimera_set = df[(df['so'] >= high_so) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        high_ho_original_set = df[(df['ho'] >= high_ho) & (df['channel'] == channel) & (df['category'] == 'original')]
        high_ho_chimera_set = df[(df['ho'] >= high_ho) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        
        low_sp_original_set = df[(df['sp'] <= low_sp) & (df['channel'] == channel) & (df['category'] == 'original')]
        low_sp_chimera_set = df[(df['sp'] <= low_sp) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        low_hp_original_set = df[(df['hp'] <= low_hp) & (df['channel'] == channel) & (df['category'] == 'original')]
        low_hp_chimera_set = df[(df['hp'] <= low_hp) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        low_so_original_set = df[(df['so'] <= low_so) & (df['channel'] == channel) & (df['category'] == 'original')]
        low_so_chimera_set = df[(df['so'] <= low_so) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        low_ho_original_set = df[(df['ho'] <= low_ho) & (df['channel'] == channel) & (df['category'] == 'original')]
        low_ho_chimera_set = df[(df['ho'] <= low_ho) & (df['channel'] == channel) & (df['category'] == 'chimera')]
        
        high_sp_ori_erp = np.squeeze(np.array(high_sp_original_set['erp'].to_list()), axis=1)
        high_sp_ori_erp_ave = np.average(high_sp_ori_erp, axis=0)
        high_hp_ori_erp = np.squeeze(np.array(high_hp_original_set['erp'].to_list()), axis=1)
        high_hp_ori_erp_ave = np.average(high_hp_ori_erp, axis=0)
        high_so_ori_erp = np.squeeze(np.array(high_so_original_set['erp'].to_list()), axis=1)
        high_so_ori_erp_ave = np.average(high_so_ori_erp, axis=0)
        high_ho_ori_erp = np.squeeze(np.array(high_ho_original_set['erp'].to_list()), axis=1)
        high_ho_ori_erp_ave = np.average(high_ho_ori_erp, axis=0)
        
        low_sp_ori_erp = np.squeeze(np.array(low_sp_original_set['erp'].to_list()), axis=1)
        low_sp_ori_erp_ave = np.average(low_sp_ori_erp, axis=0)
        low_hp_ori_erp = np.squeeze(np.array(low_hp_original_set['erp'].to_list()), axis=1)
        low_hp_ori_erp_ave = np.average(low_hp_ori_erp, axis=0)
        low_so_ori_erp = np.squeeze(np.array(low_so_original_set['erp'].to_list()), axis=1)
        low_so_ori_erp_ave = np.average(low_so_ori_erp, axis=0)
        low_ho_ori_erp = np.squeeze(np.array(low_ho_original_set['erp'].to_list()), axis=1)
        low_ho_ori_erp_ave = np.average(low_ho_ori_erp, axis=0)
        
        ERP_all_df = ERP_all_df.append({'subject':subject,'category':'original','channel':channel,
                                            'high_sp':high_sp_ori_erp_ave,
                                            'high_hp':high_hp_ori_erp_ave,
                                            'high_so':high_so_ori_erp_ave,
                                            'high_ho':high_ho_ori_erp_ave,
                                            'low_sp':low_sp_ori_erp_ave,
                                            'low_hp':low_hp_ori_erp_ave,
                                            'low_so':low_so_ori_erp_ave,
                                            'low_ho':low_ho_ori_erp_ave},ignore_index=True)
        
        high_sp_chimera_erp = np.squeeze(np.array(high_sp_chimera_set['erp'].to_list()), axis=1)
        high_sp_chimera_erp_ave = np.average(high_sp_chimera_erp, axis=0)
        high_hp_chimera_erp = np.squeeze(np.array(high_hp_chimera_set['erp'].to_list()), axis=1)
        high_hp_chimera_erp_ave = np.average(high_hp_chimera_erp, axis=0)
        high_so_chimera_erp = np.squeeze(np.array(high_so_chimera_set['erp'].to_list()), axis=1)
        high_so_chimera_erp_ave = np.average(high_so_chimera_erp, axis=0)
        high_ho_chimera_erp = np.squeeze(np.array(high_ho_chimera_set['erp'].to_list()), axis=1)
        high_ho_chimera_erp_ave = np.average(high_ho_chimera_erp, axis=0)
        
        low_sp_chimera_erp = np.squeeze(np.array(low_sp_chimera_set['erp'].to_list()), axis=1)
        low_sp_chimera_erp_ave = np.average(low_sp_chimera_erp, axis=0)
        low_hp_chimera_erp = np.squeeze(np.array(low_hp_chimera_set['erp'].to_list()), axis=1)
        low_hp_chimera_erp_ave = np.average(low_hp_chimera_erp, axis=0)
        low_so_chimera_erp = np.squeeze(np.array(low_so_chimera_set['erp'].to_list()), axis=1)
        low_so_chimera_erp_ave = np.average(low_so_chimera_erp, axis=0)
        low_ho_chimera_erp = np.squeeze(np.array(low_ho_chimera_set['erp'].to_list()), axis=1)
        low_ho_chimera_erp_ave = np.average(low_ho_chimera_erp, axis=0)
        
        ERP_all_df = ERP_all_df.append({'subject':subject,'category':'chimera','channel':channel,
                                           'high_sp':high_sp_chimera_erp_ave,
                                           'high_hp':high_hp_chimera_erp_ave,
                                           'high_so':high_so_chimera_erp_ave,
                                           'high_ho':high_ho_chimera_erp_ave,
                                           'low_sp':low_sp_chimera_erp_ave,
                                           'low_hp':low_hp_chimera_erp_ave,
                                           'low_so':low_so_chimera_erp_ave,
                                           'low_ho':low_ho_chimera_erp_ave},ignore_index=True)
        
ERP_all_df.to_pickle(exp_path+'/note_ERP_expectation_all.pkl')
# %% statistics plot
ERP_all_df = pd.read_pickle(exp_path+'/note_ERP_expectation_all.pkl')
