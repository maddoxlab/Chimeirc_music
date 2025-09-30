#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:15:04 2023

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import fft, ifft
from expyfun.io import write_hdf5, read_hdf5, read_tab
import mne
from mne.filter import resample
import matplotlib.pyplot as plt
import os
# %% Define Filtering Functions

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
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

# %% Parameters
# Anlysis
is_ABR = True # if derive only ABR
Bayesian = True # Bayesian averaging
phase_only = False # if use phase-only regressor
# Stim param
stim_fs = 48000 # stimulus sampling frequency
t_max = 27
n_epoch_total = 132
n_epoch_ori = 66
n_epoch_chimera = 66
# EEG param
eeg_n_channel = 2 # total channel of ABR
eeg_fs = 10000 # eeg sampling frequency
eeg_f_hp = 1 # high pass cutoff

#%% Subject
subject_list = ['chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025']
subject_ids = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
# %% File paths
exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA EXT/Chimera/'
exp_path = 'F://Chimera/'

corpus_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/'
regressor_path = corpus_path+"regressor/ANM.pkl" # Regressor files path
# %% ABR deriving
for subject, subject_id in zip(subject_list, subject_ids):
    print(subject)
    # Read raw eeg
    data_root = exp_path+"subject_{0:03d}".format(subject_id)
    if subject_id==3:
        eeg_vhdr = data_root+'/'+subject+'_2.vhdr'
    elif subject_id==10:
        eeg_vhdr = data_root+'/'+subject+'_3.vhdr'
    else:
        eeg_vhdr = data_root+'/'+subject+'.vhdr'
    eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
    if is_ABR:
        channels = ['A1p', 'A1n', 'A2p', 'A2n']
    eeg_raw = eeg_raw.pick_channels(channels)
    data = eeg_raw.get_data(picks=eeg_raw.ch_names[::2])
    data -= eeg_raw.get_data(picks=eeg_raw.ch_names[1::2])
    data /= 100
    info = mne.create_info(ch_names=["EP1","EP2"], sfreq=eeg_raw.info['sfreq'], ch_types='eeg')
    eeg_raw_ref = mne.io.RawArray(data, info)
    del data
    #  Read events
    events, event_dict = mne.events_from_annotations(eeg_raw)
    
    events_2trig = np.zeros((0, 3)).astype(int)
    events_new = np.zeros((1, 3)).astype(int)
    events_new[0, :] = events[1, :]
    index = []
    for i in range(len(events)-1):
        if events[i, 2] == 2:
            index += [i]
            events_2trig = np.append(events_2trig, [events[i, :]], axis=0)
            events_new = np.append(events_new, [events[i+1, :]], axis=0)
    events_2trig = np.append(events_2trig, [events[-1, :]], axis=0)
    time_diff = events_2trig[:, 0] - events_new[:, 0]
    for file in os.listdir(data_root):
        if file.endswith(".tab"):
            tab_path = data_root +"/"+file
    tab = read_tab(tab_path, group_start='Drift triggers were stamped at the folowing times: ',
                   group_end=None, return_params=False)
    time_drift_trigger = []
    preference = []
    for ei in range(n_epoch_total):
        time_drift_trigger += [float(tab[ei]['Drift triggers were stamped at the folowing times: '][0][0][1:-1])]
        preference += [int(tab[ei]['Trial Params'][0][0][1:-1].split(",")[3].split(": ")[1])]

    eeg_fs_n = np.average(time_diff / time_drift_trigger)
    #eeg_fs_n = 10000
    #  EEG Preprocessing
    print('Filtering raw EEG data...')
    # High-pass filter
    eeg_raw_ref._data = butter_highpass_filter(eeg_raw_ref._data, eeg_f_hp, eeg_fs_n)
    # Notch filter
    notch_freq = np.arange(60, 540, 120)
    notch_width = 5
    for nf in notch_freq:
        bn, an = signal.iirnotch(nf / (eeg_fs_n / 2.), float(nf) / notch_width)
        eeg_raw_ref._data = signal.lfilter(bn, an, eeg_raw_ref._data)
    # Epoch params
    # Get epoch order list
    for file in os.listdir(data_root):
        if file.endswith(".csv"):
            csv_file_path = data_root +"/"+ file
    events_df = pd.read_csv(csv_file_path)
    # Epoching
    print('Epoching EEG data...')
    epochs = mne.Epochs(eeg_raw_ref, events, event_id=1, tmin=0, tmax=t_max+1, baseline=None, preload=True)
    epochs = epochs.get_data()
    stim_types = ['ori','chimera']
    ori_epochs = events_df[events_df['type']==0]
    ori_epochs = ori_epochs.reset_index()
    chimera_epochs = events_df[events_df['type']==1]
    chimera_epochs = chimera_epochs.reset_index()
    #  Analysis
    # 
    len_eeg = int(t_max*eeg_fs)
    regressor = pd.read_pickle(regressor_path)
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    w = dict(ori=np.zeros(len_eeg),
             chimera=np.zeros(len_eeg))
    abr = dict(ori=np.zeros(8000),
             chimera=np.zeros(8000))
    abr_bp = dict(ori=np.zeros(8000),
             chimera=np.zeros(8000))
    ###### Original
    x_in_pos = np.zeros((n_epoch_ori, len_eeg))
    x_in_neg = np.zeros((n_epoch_ori, len_eeg))
    x_out = np.zeros((n_epoch_ori, len_eeg))
    # Load x_in
    for ei in range(n_epoch_ori):
        f_name = ori_epochs.loc[ei, 'name']
        # epoch_len = np.floor(ori_epochs.loc[ei, 'duration']*eeg_fs)
        reg_ind = regressor[regressor['name']==f_name].index.values.astype(int)[0]
        x_in_pos_temp = regressor.loc[reg_ind, 'ANM_pos'][0:int(len_eeg)]
        x_in_pos[ei, :] = x_in_pos_temp
        x_in_neg_temp = regressor.loc[reg_ind, 'ANM_neg'][0:int(len_eeg)]
        x_in_neg[ei, :] = x_in_neg_temp
        eeg_id = ori_epochs.loc[ei, 'index']
        x_out_temp = epochs[eeg_id]
        x_out_temp = resample(x_out_temp, down=eeg_fs_n/eeg_fs)
        x_out_temp = np.mean(x_out_temp[:,0:int(len_eeg)], axis=0)
        x_out[ei, :] = x_out_temp
        
    # x_in fft
    x_in_pos_fft = fft(x_in_pos)
    x_in_neg_fft = fft(x_in_neg)
    # x_out fft
    x_out_fft = fft(x_out)
    if Bayesian:
        ivar = 1 / np.var(x_out, axis=1)
        weight = ivar/np.nansum(ivar)
    # TRF
    denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
    denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
    w_pos = []
    w_neg = []
    for ei in range(n_epoch_ori):
        w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                  x_out_fft[ei, :]) / denom_pos
        w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                  x_out_fft[ei, :]) / denom_neg
        w_pos += [w_i_pos]
        w_neg += [w_i_neg]
    w['ori'] = (ifft(np.array(w_pos).sum(0)).real +
                  ifft(np.array(w_neg).sum(0)).real) / 2
    abr['ori'] = np.concatenate((w['ori'][int(t_start*eeg_fs):],
                                    w['ori'][0:int(t_stop*eeg_fs)]))
    # shift ABR for ANM regressor
    abr['ori']  = np.roll(abr['ori'] , int(2.75*eeg_fs/1000))
    abr_bp['ori'] = butter_bandpass_filter(abr['ori'], 1, 1500, eeg_fs)
    
    ###### For chimera response
    len_eeg = int(t_max*eeg_fs)
    regressor = pd.read_pickle(regressor_path)
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    # Original
    x_in_pos = np.zeros((n_epoch_ori, len_eeg))
    x_in_neg = np.zeros((n_epoch_ori, len_eeg))
    x_out = np.zeros((n_epoch_ori, len_eeg))
    # Load x_in
    for ei in range(n_epoch_chimera):
        f_name = chimera_epochs.loc[ei, 'name']
        # epoch_len = np.floor(ori_epochs.loc[ei, 'duration']*eeg_fs)
        reg_ind = regressor[regressor['name']==f_name].index.values.astype(int)[0]
        x_in_pos_temp = regressor.loc[reg_ind, 'ANM_pos'][0:int(len_eeg)]
        x_in_pos[ei, :] = x_in_pos_temp
        x_in_neg_temp = regressor.loc[reg_ind, 'ANM_neg'][0:int(len_eeg)]
        x_in_neg[ei, :] = x_in_neg_temp
        eeg_id = chimera_epochs.loc[ei, 'index']
        x_out_temp = epochs[eeg_id]
        x_out_temp = resample(x_out_temp, down=eeg_fs_n/eeg_fs)
        x_out_temp = np.mean(x_out_temp[:,0:int(len_eeg)], axis=0)
        x_out[ei, :] = x_out_temp
        
    # x_in fft
    x_in_pos_fft = fft(x_in_pos)
    x_in_neg_fft = fft(x_in_neg)
    # x_out fft
    x_out_fft = fft(x_out)
    if Bayesian:
        ivar = 1 / np.var(x_out, axis=1)
        weight = ivar/np.nansum(ivar)
    # TRF
    denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
    denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
    w_pos = []
    w_neg = []
    for ei in range(n_epoch_ori):
        w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                  x_out_fft[ei, :]) / denom_pos
        w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                  x_out_fft[ei, :]) / denom_neg
        w_pos += [w_i_pos]
        w_neg += [w_i_neg]
    w['chimera'] = (ifft(np.array(w_pos).sum(0)).real +
                  ifft(np.array(w_neg).sum(0)).real) / 2
    abr['chimera'] = np.concatenate((w['chimera'][int(t_start*eeg_fs):],
                                     w['chimera'][0:int(t_stop*eeg_fs)]))
    # shift ABR for ANM regressor
    abr['chimera']  = np.roll(abr['chimera'] , int(2.75*eeg_fs/1000))
    abr_bp['chimera'] = butter_bandpass_filter(abr['chimera'], 1, 1500, eeg_fs)

    # plt.plot(lags, abr['ori'])
    # plt.plot(lags, abr['chimera'])
    # plt.plot(lags, (abr['ori']+abr['chimera'])/2)
    # plt.xlim(-10, 30)
    # plt.grid()
    # plt.legend(["original","chimera","averaged"])
    # plt.title(subject)
    
    write_hdf5(data_root+'/'+subject+'_ABR_data.hdf5',
               dict(w=w, abr=abr, abr_bp=abr_bp, lags=lags), overwrite=True)


# %% AVERAGE ABR
from scipy import stats
from statsmodels.stats import multitest

subject_list = ['chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025']
subject_ids = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]

subject_list = ['chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025']
subject_ids = [2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25]


ABR_ori = np.zeros((len(subject_list),8000))
ABR_chimera = np.zeros((len(subject_list),8000))
for i, subject in enumerate(subject_list):
    data_root = exp_path+'subject'+subject[7:]
    ABR_data = read_hdf5(data_root+'/'+subject+'_ABR_data.hdf5')
    ABR_ori[i] = ABR_data['abr_bp']['ori']
    ABR_chimera[i] = ABR_data['abr_bp']['chimera']
    lags = ABR_data["lags"]

ABR_ori_ave = np.average(ABR_ori, axis=0)
ABR_ori_se = np.std(ABR_ori, axis=0)/np.sqrt(len(subject_list))
ABR_chimera_ave = np.average(ABR_chimera, axis=0)
ABR_chimera_se = np.std(ABR_chimera, axis=0)/np.sqrt(len(subject_list))

ABR_ori_vs_chimera = ABR_ori-ABR_chimera
ABR_ori_vs_chimera_ave = np.average(ABR_ori_vs_chimera, axis=0)
ABR_ori_vs_chimera_se = np.std(ABR_ori_vs_chimera, axis=0)/np.sqrt(len(subject_list))

# Weight Stats
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
alpha=0.05
res = stats.permutation_test((ABR_ori[:,2000:2151], ABR_chimera[:,2000:2151]),statistic,permutation_type='samples',axis=0)
p_raw = res.pvalue
indices = np.where(p_raw<=alpha)[0]
sig_time_corrected, p_corrected = multitest.fdrcorrection(res.pvalue,alpha=alpha)
        
# %% Plot
dpi = 300
figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/poster/'
figure_path = 'D://AMPLab/MusicExp/paper_figs/'


plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, ABR_ori_ave, c="C0", linewidth=2, linestyle='solid', label="Original")
plt.fill_between(lags, ABR_ori_ave-ABR_ori_se, ABR_ori_ave+ABR_ori_se, color="C0", alpha=0.8)
plt.plot(lags, ABR_chimera_ave, c="C1", linewidth=1, linestyle='dashed', label="Chimera")
plt.fill_between(lags, ABR_chimera_ave-ABR_chimera_se, ABR_chimera_ave+ABR_chimera_se, color="C1", alpha=0.5)
plt.xlim(-10, 30)
plt.ylim(-40, 65)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.grid()
plt.legend(fontsize=24)
plt.tight_layout()
plt.savefig(figure_path+'ABR_results.png', dpi=dpi, format='png')


plt.plot(lags, ABR_ori_vs_chimera_ave, c="C0", linewidth=2, linestyle='solid', label="Diff")
plt.fill_between(lags, ABR_ori_vs_chimera_ave-ABR_ori_vs_chimera_se, ABR_ori_vs_chimera_ave+ABR_ori_vs_chimera_se, color="C0", alpha=0.8)
plt.xlim(-10, 30)
plt.ylim(-40, 65)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.grid()
plt.tight_layout()


# Split by musicianship
formal_training_years = [10,1.5,5,1.5,0,3,1,2,10,5,7,7,0,0,1,11,5,10,0,2,10,5,2]
# Order of musicianship accend
order_i = [i for i, value in sorted(enumerate(formal_training_years),key=lambda x:x[1], reverse=False)]

nonmusician_ind = [i for i in (np.where(np.array(formal_training_years)<1.5)[0])]
musician_ind = [i for i in (np.where(np.array(formal_training_years)>=6.75)[0])]

subject_list_nonmusician = [subject_list[i] for i in (np.where(np.array(formal_training_years)<3.5)[0])]
subject_list_musician = [subject_list[i] for i in (np.where(np.array(formal_training_years)>=3.5)[0])]

ABR_ori_non_ave = np.mean(ABR_ori[nonmusician_ind], axis=0)
ABR_ori_non_se = np.std(ABR_ori[nonmusician_ind], axis=0)/np.sqrt(len(subject_list_nonmusician))
ABR_ori_mus_ave = np.mean(ABR_ori[musician_ind], axis=0)
ABR_ori_mus_se = np.std(ABR_ori[musician_ind], axis=0)/np.sqrt(len(subject_list_musician))

ABR_chimera_non_ave = np.mean(ABR_chimera[nonmusician_ind], axis=0)
ABR_chimera_non_se = np.std(ABR_chimera[nonmusician_ind], axis=0)/np.sqrt(len(subject_list_nonmusician))
ABR_chimera_mus_ave = np.mean(ABR_chimera[musician_ind], axis=0)
ABR_chimera_mus_se = np.std(ABR_chimera[musician_ind], axis=0)/np.sqrt(len(subject_list_musician))

fig = plt.figure(dpi=dpi)
fig.set_size_inches(5, 4)
plt.plot(lags, ABR_ori_mus_ave, c="C0", linewidth=2, linestyle='solid', label="Original-Musician")
plt.fill_between(lags, ABR_ori_mus_ave-ABR_ori_mus_se, ABR_ori_mus_ave+ABR_ori_mus_se, color="C0", alpha=0.8)
plt.plot(lags, ABR_ori_non_ave, c="k", linewidth=2, linestyle='solid', label="Original-Nonmusician")
plt.fill_between(lags, ABR_ori_non_ave-ABR_ori_non_se, ABR_ori_non_ave+ABR_ori_non_se, color="k", alpha=0.5)
plt.xlim(-10, 30)
plt.ylim(-40, 65)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()

fig = plt.figure(dpi=dpi)
fig.set_size_inches(5, 4)
plt.plot(lags, ABR_chimera_mus_ave, c="C1", linewidth=2, linestyle='dashed', label="Chimeric-Musician")
plt.fill_between(lags, ABR_chimera_mus_ave-ABR_chimera_mus_se, ABR_chimera_mus_ave+ABR_chimera_mus_se, color="C1", alpha=0.8)
plt.plot(lags, ABR_chimera_non_ave, c="k", linewidth=2, linestyle='dashed', label="Chimeric-Nonmusician")
plt.fill_between(lags, ABR_chimera_non_ave-ABR_chimera_non_se, ABR_chimera_non_ave+ABR_chimera_non_se, color="k", alpha=0.5)
plt.xlim(-10, 30)
plt.ylim(-40, 65)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()

# %% Plot by individual vertex
ABR_ori = np.zeros((len(subject_list),8000))
ABR_chimera = np.zeros((len(subject_list),8000))
ABR_averaged = np.zeros((len(subject_list),8000))
for i, subject in enumerate(subject_list):
    data_root = exp_path+'subject'+subject[7:]
    ABR_data = read_hdf5(data_root+'/'+subject+'_ABR_data.hdf5')
    ori_temp = ABR_data['abr']['ori']
    ABR_ori[i] = butter_bandpass_filter(ori_temp, 1, 1500, eeg_fs,order=2)
    chimera_temp = ABR_data['abr']['chimera']
    ABR_chimera[i] = butter_bandpass_filter(chimera_temp, 1, 1500, eeg_fs,order=2)
    ABR_averaged[i] = (ABR_ori[i] + ABR_chimera[i])/2
    lags = ABR_data["lags"]

ABR_ori_ave = np.average(ABR_ori, axis=0)
ABR_ori_se = np.std(ABR_ori, axis=0)/np.sqrt(len(subject_list))
ABR_chimera_ave = np.average(ABR_chimera, axis=0)
ABR_chimera_se = np.std(ABR_chimera, axis=0)/np.sqrt(len(subject_list))

fig, axes=plt.subplots(1,3,figsize=(15,30),sharey=True)
fig.set_size_inches(15, 30)
offset = 120
for k in order_i:
    axes[0].plot(lags, ABR_ori[k]+k*offset, c="C0", linewidth=2, linestyle='solid')
for k in order_i:
    axes[1].plot(lags, ABR_chimera[k]+k*offset, c="C1", linewidth=2, linestyle='solid')
for k in order_i:
    axes[2].plot(lags, ABR_averaged[k]+k*offset, c="k", linewidth=2, linestyle='solid')
axes[0].set_title('Original')
axes[0].set_xlim(-10, 30)
axes[0].grid(True)
axes[1].set_title('Chimeric')
axes[1].set_xlim(-10, 30)
axes[1].grid(True)
axes[2].set_title('Averaged')
axes[2].set_xlim(-10, 30)
axes[2].grid(True)
plt.savefig(figure_path+'ABR_results_musician_order.png', dpi=dpi, format='png')


