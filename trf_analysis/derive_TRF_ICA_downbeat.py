#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 13:49:04 2023

@author: tshan@urmc-sh.rochester.edu
"""
import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from expyfun.io import write_hdf5, read_hdf5
from mtrf.model import TRF
from mtrf.stats import crossval
from statsmodels.stats import multitest
import mne
#import cupy
import matplotlib.pyplot as plt

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
# %% 
subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

# %%
for subject in subject_list:
    print(subject)
    onset_cut = 1
    do_zscore = False
    exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA EXT/Chimera/'
    regressor_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/125hz/'
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    
    # Load set pair value data
    set_dic = read_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5')
    # Load regressors
    downbeat_df = pd.read_pickle(regressor_path+'downbeat_trf_reg.pkl')
    
    # Load EEG data
    eeg_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
    epoch_ori = eeg_data['epoch_ori']
    ori_epochs_df = eeg_data['ori_epochs_df']
    epoch_chimera = eeg_data['epoch_chimera']
    chimera_epochs_df = eeg_data['chimera_epochs_df']
    
    # for ei in range(n_epoch_ori):
    #     std = np.std(epoch_ori[ei, 17, :])
    #     print(ei, std)
        
        
    ##### Analysis Original
    print("Original")
    # Trim epochs and concatenate epochs
    x_in_ori_trim = [] # onset, downbeat (pitch == time)
    x_out_ori_trim = []
    f_name_ori = []
    set_num_ori = []
    for ei in range(n_epoch_ori):
        f_name = ori_epochs_df.loc[ei, 'name']
        f_name_ori += [f_name[:-4]]
        set_num_ori += [set_dic[f_name_ori[-1]]]
        # cut onset for both regressor and eeg data
        # Get x_in
        # Downbeat regressors 
        db_ind = downbeat_df[downbeat_df['name']==f_name].index.values.astype(int)[0]
        epoch_len = np.floor(downbeat_df.loc[db_ind, 'duration']*eeg_fs_down)
        onset_x_in_temp = downbeat_df.loc[db_ind, 'onset'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        downbeat_both_x_in_temp = downbeat_df.loc[db_ind, 'downbeat_both'][int(onset_cut*eeg_fs_down):int(epoch_len)]

        x_in_temp = np.array([onset_x_in_temp,downbeat_both_x_in_temp]) 
        x_in_ori_trim += [x_in_temp.T] # Transpose to match mTRF toolbox dimension
        # Get x_out
        x_out_temp = epoch_ori[ei,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        #x_out_temp = np.delete(x_out_temp,17, axis=0)
        x_out_ori_trim += [x_out_temp.T] # Transpose to match mTRF toolbox dimension
    
    if subject == 'chimera_012':
        del x_in_ori_trim[45:47]
        del x_out_ori_trim[45:47]
    # TRF training
    trf_ori = TRF(direction=1)
    reg=0
    trf_ori.train(x_in_ori_trim, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=reg)
    trf_ori.save(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0_trfobj_ori.pkl')
    # Cross validation
    r_ori_fwd = crossval(trf_ori, x_in_ori_trim, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=reg, k=10)
    

    ##### Analysis Chimera
    print("Chimera")
    # Trim epochs and concatenate epochs
    x_in_chimera_trim = [] # onset, downbeat (pitch == time)
    x_out_chimera_trim = []
    f_name_chimera = []
    set_num_chimera = []
    for ei in range(n_epoch_chimera):
        f_name = chimera_epochs_df.loc[ei, 'name']
        f_name_chimera += [f_name[:-4]]
        set_num_chimera += [set_dic[f_name_chimera[-1]]]
        # cut onset for both regressor and eeg data
        # Get x_in
        # Downbeat regressors 
        db_ind = downbeat_df[downbeat_df['name']==f_name].index.values.astype(int)[0]
        epoch_len = np.floor(downbeat_df.loc[db_ind, 'duration']*eeg_fs_down)
        onset_x_in_temp = downbeat_df.loc[db_ind, 'onset'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        downbeat_pitch_x_in_temp = downbeat_df.loc[db_ind, 'downbeat_pitch'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        downbeat_onset_x_in_temp = downbeat_df.loc[db_ind, 'downbeat_onset'][int(onset_cut*eeg_fs_down):int(epoch_len)]
        downbeat_both_x_in_temp = downbeat_df.loc[db_ind, 'downbeat_both'][int(onset_cut*eeg_fs_down):int(epoch_len)]

        x_in_temp = np.array([onset_x_in_temp,downbeat_pitch_x_in_temp,downbeat_onset_x_in_temp,downbeat_both_x_in_temp]) 
        x_in_chimera_trim += [x_in_temp.T] # Transpose to match mTRF toolbox dimension
        # Get x_out
        x_out_temp = epoch_chimera[ei,:,int(onset_cut*eeg_fs_down):int(epoch_len)]
        #x_out_temp = np.delete(x_out_temp,17, axis=0)
        x_out_chimera_trim += [x_out_temp.T] # Transpose to match mTRF toolbox dimension
        
    # TRF training
    trf_chimera = TRF(direction=1)
    reg=0
    trf_chimera.train(x_in_chimera_trim, x_out_chimera_trim, eeg_fs_down, -0.1, 0.7, regularization=reg)
    trf_chimera.save(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0_trfobj_chimera.pkl')
    # Cross validation
    r_chimera_fwd = crossval(trf_chimera, x_in_ori_trim, x_out_ori_trim, eeg_fs_down, -0.1, 0.7, regularization=reg, k=10)
    
    # Plot Fz
    # plt.plot(trf_ori.times, trf_ori.weights[0,:,1],label='ori onset')
    # plt.plot(trf_ori.times, trf_ori.weights[1,:,1],label='ori db')
    # plt.plot(trf_chimera.times, trf_chimera.weights[0,:,1],label='chimera onset')
    # plt.plot(trf_chimera.times, trf_chimera.weights[1,:,1],label='chimera db pitch')
    # plt.plot(trf_chimera.times, trf_chimera.weights[2,:,1],label='chimera db onset')
    # plt.plot(trf_chimera.times, trf_chimera.weights[3,:,1],label='chimera db')
    # plt.legend()
    # plt.grid()
    
    write_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5',
               dict(ori_weights= trf_ori.weights,
                    chimera_weights=trf_chimera.weights,
                    time=trf_ori.times,
                    reg=reg,
                    set_num_ori=set_num_ori,
                    set_num_chimera=set_num_chimera),
               overwrite=True)

# %% TRF all analysis
subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011', 'chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021', 'chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

subject_num = len(subject_list)
# Load data
trf_ori_all = np.zeros((subject_num,2,102,32))
trf_chimera_all = np.zeros((subject_num,4,102,32))

for i, subject in enumerate(subject_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5')
    trf_ori_all[i] = trf_data['ori_weights']
    trf_chimera_all[i] = trf_data['chimera_weights']
    lag = trf_data['time']*1000

trf_chimera_downbeat_3 = np.sum(trf_chimera_all[:,1:4,:,:],axis=1,keepdims=True)
trf_chimera_downbeat_2 = np.sum(trf_chimera_all[:,1:3,:,:],axis=1,keepdims=True)
trf_chimera_all = np.concatenate((trf_chimera_all, trf_chimera_downbeat_3, trf_chimera_downbeat_2), axis=1)

trf_ori_all_ave = np.average(trf_ori_all, axis=0)
trf_ori_all_err = np.std(trf_ori_all, axis=0)/np.sqrt(subject_num)
trf_chimera_all_ave = np.average(trf_chimera_all, axis=0)
trf_chimera_all_err = np.std(trf_chimera_all, axis=0)/np.sqrt(subject_num)

# %% Plot
from matplotlib.ticker import ScalarFormatter
dpi = 300
data_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicABR/MusicExp/'
figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/poster/'

plt.rc('axes', titlesize=20, labelsize=20)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)


chn=1
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
# Original
plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch")
plt.fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Downbeat timing")
plt.fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch*timing")
plt.fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.grid()
plt.title(channel_names[chn])
#plt.tight_layout()
lg = plt.legend(fontsize=24, bbox_to_anchor=(1.03, 1.0))
plt.show()
plt.savefig(figure_path+'TRF_downbeat.png', dpi=dpi, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')

# Plot
chn=1
plt.figure()
# Original
# plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
# plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
# plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
# plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[5,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat Addded")
plt.fill_between(lag, trf_chimera_all_ave[5,:,chn]-trf_chimera_all_err[5,:,chn], trf_chimera_all_ave[5,:,chn]+trf_chimera_all_err[5,:,chn], color='C1',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title(channel_names[chn])
plt.legend()
plt.show()


# %% Averaged Topos
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(channel_names, sfreq=eeg_fs_down, ch_types='eeg')

# Original onset
original_onset_trf = mne.EvokedArray(trf_ori_all_ave[0].T,info, tmin=-0.1, kind="average")
original_onset_trf.set_montage(montage)
original_onset_trf.nave = len(subject_list)
original_onset_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-2e-4, 4e-4]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-4e-4, 4e-4), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Note onset")
# Original downbeat
original_downbeat_trf = mne.EvokedArray(trf_ori_all_ave[1].T,info, tmin=-0.1, kind="average")
original_downbeat_trf.set_montage(montage)
original_downbeat_trf.nave = len(subject_list)
original_downbeat_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-6.5e-5, 6.5e-5]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-6.5e-5, 6.5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Downbeat")
# Chimera onset
chimera_onset_trf = mne.EvokedArray(trf_chimera_all_ave[0].T,info, tmin=-0.1, kind="average")
chimera_onset_trf.set_montage(montage)
chimera_onset_trf.nave = len(subject_list)
chimera_onset_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-2e-4, 4e-4]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-4e-4, 4e-4), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Note onset")
# Chimera pich 
chimera_db_pitch_trf = mne.EvokedArray(trf_chimera_all_ave[1].T,info, tmin=-0.1, kind="average")
chimera_db_pitch_trf.set_montage(montage)
chimera_db_pitch_trf.nave = len(subject_list)
chimera_db_pitch_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-6.5e-5, 6.5e-5]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-6.5e-5, 6.5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Downbeat pitch")
# Chimera timing
chimera_db_onset_trf = mne.EvokedArray(trf_chimera_all_ave[2].T,info, tmin=-0.1, kind="average")
chimera_db_onset_trf.set_montage(montage)
chimera_db_onset_trf.nave = len(subject_list)
chimera_db_onset_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-6.5e-5, 6.5e-5]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-6.5e-5, 6.5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Downbeat timing")
# Chimera interaction
chimera_db_both_trf = mne.EvokedArray(trf_chimera_all_ave[3].T,info, tmin=-0.1, kind="average")
chimera_db_both_trf.set_montage(montage)
chimera_db_both_trf.nave = len(subject_list)
chimera_db_both_trf.plot_joint(times = [60e-3, 110e-3,160e-3,210e-3,260e-3,310e-3,360e-3,410e-3,460e-3,510e-3,560e-3,610e-3,660e-3],
                            ts_args=dict(ylim=dict(eeg=[-6.5e-5, 6.5e-5]), xlim=[-100, 700],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-6.5e-5, 6.5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Downbeat pitch*timing")

# %% Cluster statistical test
from scipy.stats import ttest_ind
import mne
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test

plt.rc('axes', titlesize=24, labelsize=24)
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)

montage = mne.channels.make_standard_montage("easycap-M1")
info = mne.create_info(channel_names, sfreq=eeg_fs_down,ch_types='eeg')
info.set_montage(montage)

###### Original note onset
trf_ori_onset = trf_ori_all[:,0,:,:]
trf_ori_onset = np.swapaxes(trf_ori_onset, 1, 2)
Epochs_ori_onset = mne.EpochsArray(trf_ori_onset, info, tmin=-0.1)
Epochs_ori_onset.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_ori_onset.info, "eeg")

X_ori_onset = Epochs_ori_onset.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_ori_onset, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_ori_onset = Epochs_ori_onset.average()
time_unit = dict(time_unit="ms")
evoked_ori_onset.plot_joint(
    title="Original Downbeat", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_ori_onset.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_ori_onset.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=ori_onset_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()


###### Original downbeat
trf_ori_downbeat = trf_ori_all[:,1,:,:]
trf_ori_downbeat = np.swapaxes(trf_ori_downbeat, 1, 2)
Epochs_ori_downbeat = mne.EpochsArray(trf_ori_downbeat, info, tmin=-0.1)
Epochs_ori_downbeat.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_ori_downbeat.info, "eeg")

X_ori_downbeat = Epochs_ori_downbeat.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_ori_downbeat, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_downbeat_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_downbeat_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_ori_downbeat = Epochs_ori_downbeat.average()
time_unit = dict(time_unit="ms")
evoked_ori_downbeat.plot_joint(
    title="Original Downbeat", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_ori_downbeat.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(10, 12))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_ori_downbeat.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=ori_downbeat_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
axes['Left'].set_title("Left")
axes['Midline'].set_title("Midline")
axes['Right'].set_title("Right")
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"original_downbeat.png",format='png',dpi=dpi)

###### Chimera onset
trf_chimera_onset = trf_chimera_all[:,0,:,:]
trf_chimera_onset = np.swapaxes(trf_chimera_onset, 1, 2)
Epochs_chimera_onset = mne.EpochsArray(trf_chimera_onset, info, tmin=-0.1)
Epochs_chimera_onset.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_chimera_onset.info, "eeg")

X_chimera_onset = Epochs_chimera_onset.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_chimera_onset, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
chimera_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(chimera_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_chimera_onset = Epochs_chimera_onset.average()
time_unit = dict(time_unit="ms")
evoked_chimera_onset.plot_joint(
    title="Chimera onset", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_chimera_onset.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_chimera_onset.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=chimera_onset_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()

######## Originl vs Chimera onset
X_ori_vs_chimera_onset = [Epochs_ori_onset.get_data().transpose(0,2,1),
     Epochs_chimera_onset.get_data().transpose(0,2,1),]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X_ori_vs_chimera_onset, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_vs_chimera_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(chimera_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_ori_vs_chimera_onset = mne.combine_evoked([Epochs_ori_onset.average(), Epochs_chimera_onset.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked_ori_vs_chimera_onset.plot_joint(
    title="Original vs Chimera onset", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_ori_vs_chimera_onset.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_ori_vs_chimera_onset.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=ori_vs_chimera_onset_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()



###### Chimera pitch
trf_chimera_pitch = trf_chimera_all[:,1,:,:]
trf_chimera_pitch = np.swapaxes(trf_chimera_pitch, 1, 2)
Epochs_chimera_pitch = mne.EpochsArray(trf_chimera_pitch, info, tmin=-0.1)
Epochs_chimera_pitch.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_chimera_pitch.info, "eeg")

X_chimera_pitch = Epochs_chimera_pitch.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_chimera_pitch, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
chimera_pitch_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(chimera_pitch_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_chimera_pitch = Epochs_chimera_pitch.average()
time_unit = dict(time_unit="ms")
evoked_chimera_pitch.plot_joint(
    title="Chimera Pitch", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_chimera_pitch.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_chimera_pitch.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=chimera_pitch_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()

###### Chimera timing
trf_chimera_timing = trf_chimera_all[:,2,:,:]
trf_chimera_timing = np.swapaxes(trf_chimera_timing, 1, 2)
Epochs_chimera_timing = mne.EpochsArray(trf_chimera_timing, info, tmin=-0.1)
Epochs_chimera_timing.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_chimera_timing.info, "eeg")

X_chimera_timing = Epochs_chimera_timing.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_chimera_timing, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
chimera_timing_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(chimera_timing_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_chimera_timing = Epochs_chimera_timing.average()
time_unit = dict(time_unit="ms")
evoked_chimera_timing.plot_joint(
    title="Chimera Timing", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_chimera_timing.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(10, 12))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_chimera_timing.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=chimera_timing_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
axes['Left'].set_title("Left")
axes['Midline'].set_title("Midline")
axes['Right'].set_title("Right")
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"Chimera_timing_downbeat.png", dpi=dpi, format='png')

###### Chimera pitch*timing Interaction
trf_chimera_inter = trf_chimera_all[:,3,:,:]
trf_chimera_inter = np.swapaxes(trf_chimera_inter, 1, 2)
Epochs_chimera_inter = mne.EpochsArray(trf_chimera_inter, info, tmin=-0.1)
Epochs_chimera_inter.set_montage(montage)
adjacency, _ = find_ch_adjacency(Epochs_chimera_inter.info, "eeg")

X_chimera_inter = Epochs_chimera_inter.get_data().transpose(0,2,1)

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_1samp_test(
    X_chimera_inter, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
chimera_inter_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(chimera_inter_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_chimera_inter = Epochs_chimera_inter.average()
time_unit = dict(time_unit="ms")
evoked_chimera_inter.plot_joint(
    title="Chimera Interaction", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_chimera_inter.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(10, 12))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_chimera_inter.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=chimera_inter_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
axes['Left'].set_title("Left")
axes['Midline'].set_title("Midline")
axes['Right'].set_title("Right")
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"Chimera_interaction_downbeat.png", dpi=dpi, format='png')

# %% Stacked TRF
from matplotlib.ticker import FixedLocator, FixedFormatter
chn=1
fig, axes = plt.subplots(6, sharex=True)
fig.set_size_inches(7,16)
axes[0].plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
axes[0].fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
axes[0].set_ylim(-2.2e-4,4.5e-4)
axes[0].fill_between(lag,-2.2e-4,4.5e-4, where=(ori_onset_significant_points[chn]==True), color='gray', alpha=0.3)
axes[0].vlines(0,-2.2e-4,4.5e-4,color='k',linestyle="solid")
axes[0].vlines(185,-2.2e-4,4.5e-4,color='gray',linestyle="dashed")
axes[0].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))

axes[1].plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
axes[1].fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
axes[1].set_ylim(-1e-4,1e-4)
axes[1].fill_between(lag,-1e-4,1e-4, where=(ori_downbeat_significant_points[chn]==True), color='gray', alpha=0.3)
axes[1].vlines(0,-1e-4,1e-4,color='k',linestyle="solid")
axes[1].vlines(185,-1e-4,1e-4,color='gray',linestyle="dashed")
axes[1].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))
#Chimera
axes[2].plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
axes[2].fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
axes[2].set_ylim(-2.2e-4,4.5e-4)
axes[2].fill_between(lag,-2.2e-4,4.5e-4, where=(chimera_onset_significant_points[chn]==True), color='gray', alpha=0.3)
axes[2].vlines(0,-2.2e-4,4.5e-4,color='k',linestyle="solid")
axes[2].vlines(185,-2.2e-4,4.5e-4,color='gray',linestyle="dashed")
axes[2].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))

axes[3].plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Pitch")
axes[3].fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
axes[3].set_ylim(-1e-4,1e-4)
axes[3].fill_between(lag,-1e-4,1e-4, where=(chimera_pitch_significant_points[chn]==True), color='gray', alpha=0.3)
axes[3].vlines(0,-1e-4,1e-4,color='k',linestyle="solid")
axes[3].vlines(185,-1e-4,1e-4,color='gray',linestyle="dashed")
axes[3].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))

axes[4].plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Timing")
axes[4].fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
axes[4].set_ylim(-1e-4,1e-4)
axes[4].fill_between(lag,-1e-4,1e-4, where=(chimera_timing_significant_points[chn]==True), color='gray', alpha=0.3)
axes[4].vlines(0,-1e-4,1e-4,color='k',linestyle="solid")
axes[4].vlines(185,-1e-4,1e-4,color='gray',linestyle="dashed")
axes[4].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))

axes[5].plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Interaction")
axes[5].fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
axes[5].set_ylim(-1e-4,1e-4)
axes[5].fill_between(lag,-1e-4,1e-4, where=(chimera_inter_significant_points[chn]==True), color='gray', alpha=0.3)
axes[5].vlines(0,-1e-4,1e-4,color='k',linestyle="solid")
axes[5].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))
axes[5].vlines(185,-1e-4,1e-4,color='gray',linestyle="dashed")
axes[5].set_xlabel("Time (ms)", fontsize=12)
for ax in axes:
    ax.hlines(0, -104, 704, color='k')
    ax.spines[['left','right','top','bottom']].set_visible(False)
    #ax.yaxis.set_visible(False)
    ax.tick_params(axis='y', which='major', labelsize=13)
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    #ax.legend(fontsize=10)
plt.show()
plt.savefig(figure_path+'downbeat_TRF_stack.svg', dpi=dpi, format='svg')

# %% Stack topos @ 185 ms
# Original Onset
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_ori_onset.plot_topomap(times=[0.185], vlim=(-4.5, 4.5), scalings=1e4, units='AU', time_unit='ms',
                                    mask=ori_onset_significant_points, contours=0,
                                    mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"Original_onset_topo.png",dpi=dpi,format='png', bbox_inches='tight')

# Original downbeat
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_ori_downbeat.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=ori_downbeat_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"Original_downbeat_topo.png",dpi=dpi,format='png', bbox_inches='tight')

# Chimera Onset
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_onset.plot_topomap(times=[0.185], vlim=(-4.5, 4.5), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_onset_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_onset_topo.png",dpi=dpi,format='png', bbox_inches='tight')

# Chimera pitch
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_pitch.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_pitch_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_pitch_topo.png",dpi=dpi,format='png', bbox_inches='tight')

# Chimera timing
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_timing.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_timing_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_timing_topo.png",dpi=dpi,format='png', bbox_inches='tight')


# Chimera interaction
fig.set_size_inches(4,4)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_inter.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_inter_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_inter_topo.png",dpi=dpi,format='png', bbox_inches='tight')



# %% Musicianship correlation 
subject_index = np.arange(0,27,1)
music_training_years = [8,23,1,5,12,0,3,1,2,20,5,3.5,7,7,0,0,1,11,10,10,0,8,14,14,10,14,6]
formal_training_years = [4,10,1.5,5,1.5,0,3,1,2,10,5,2,7,7,0,0,1,11,5,10,0,2,10,5,2,6,2]

np.median(formal_training_years)
np.quantile(formal_training_years, 0.75)


subject_list_sort = [x for _, x in sorted(zip(music_training_years, subject_list))]
subject_index_sort = [x for _, x in sorted(zip(formal_training_years, subject_index))]

# training years distribution
plt.figure()
plt.hist(formal_training_years,bins=24)
plt.show()

# Musicianship split
group_1_ind = np.where(np.array(formal_training_years)<=1.5)
group_1_list = [subject_list[x] for x in group_1_ind[0]]
group_2_ind = np.where((np.array(formal_training_years)<=3)&(np.array(formal_training_years)>1.5))
group_2_list = [subject_list[x] for x in group_2_ind[0]]
group_3_ind = np.where((np.array(formal_training_years)<=6.5)&(np.array(formal_training_years)>3))
group_3_list = [subject_list[x] for x in group_3_ind[0]]
group_4_ind = np.where(np.array(formal_training_years)>6.5)
group_4_list = [subject_list[x] for x in group_4_ind[0]]

# Group 1
subject_num = len(group_1_list)
# Load data
trf_ori_all = np.zeros((subject_num,2,102,32))
trf_chimera_all = np.zeros((subject_num,4,102,32))

for i, subject in enumerate(group_1_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5')
    trf_ori_all[i] = trf_data['ori_weights']
    trf_chimera_all[i] = trf_data['chimera_weights']
    lag = trf_data['time']

trf_ori_all_ave = np.average(trf_ori_all, axis=0)
trf_ori_all_err = np.std(trf_ori_all, axis=0)/np.sqrt(subject_num)
trf_chimera_all_ave = np.average(trf_chimera_all, axis=0)
trf_chimera_all_err = np.std(trf_chimera_all, axis=0)/np.sqrt(subject_num)

# Plot
chn=1
plt.figure()
# Original
plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch")
plt.fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Downbeat timing")
plt.fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch*timing")
plt.fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title("Group 1 \n channel: " + channel_names[chn])
plt.legend()
plt.ylim(-0.0002, 0.0005)
plt.show()

# Group 2
subject_num = len(group_2_list)
# Load data
trf_ori_all = np.zeros((subject_num,2,102,32))
trf_chimera_all = np.zeros((subject_num,4,102,32))

for i, subject in enumerate(group_2_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5')
    trf_ori_all[i] = trf_data['ori_weights']
    trf_chimera_all[i] = trf_data['chimera_weights']
    lag = trf_data['time']

trf_ori_all_ave = np.average(trf_ori_all, axis=0)
trf_ori_all_err = np.std(trf_ori_all, axis=0)/np.sqrt(subject_num)
trf_chimera_all_ave = np.average(trf_chimera_all, axis=0)
trf_chimera_all_err = np.std(trf_chimera_all, axis=0)/np.sqrt(subject_num)

# Plot
chn=1
plt.figure()
# Original
plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch")
plt.fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Downbeat timing")
plt.fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch*timing")
plt.fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title("Group 2 \n channel: " + channel_names[chn])
plt.legend()
plt.ylim(-0.0002, 0.0005)
plt.show()

# Group 3
subject_num = len(group_3_list)
# Load data
trf_ori_all = np.zeros((subject_num,2,102,32))
trf_chimera_all = np.zeros((subject_num,4,102,32))

for i, subject in enumerate(group_3_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5')
    trf_ori_all[i] = trf_data['ori_weights']
    trf_chimera_all[i] = trf_data['chimera_weights']
    lag = trf_data['time']

trf_ori_all_ave = np.average(trf_ori_all, axis=0)
trf_ori_all_err = np.std(trf_ori_all, axis=0)/np.sqrt(subject_num)
trf_chimera_all_ave = np.average(trf_chimera_all, axis=0)
trf_chimera_all_err = np.std(trf_chimera_all, axis=0)/np.sqrt(subject_num)

# Plot
chn=1
plt.figure()
# Original
plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch")
plt.fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Downbeat timing")
plt.fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch*timing")
plt.fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title("Group 3 \n channel: " + channel_names[chn])
plt.legend()
plt.ylim(-0.0002, 0.0005)
plt.show()


#Group 4
subject_num = len(group_4_list)
# Load data
trf_ori_all = np.zeros((subject_num,2,102,32))
trf_chimera_all = np.zeros((subject_num,4,102,32))

for i, subject in enumerate(group_4_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_hp1_lp8_ICA_trf_data_-01_07_downbeat_reg0.hdf5')
    trf_ori_all[i] = trf_data['ori_weights']
    trf_chimera_all[i] = trf_data['chimera_weights']
    lag = trf_data['time']

trf_ori_all_ave = np.average(trf_ori_all, axis=0)
trf_ori_all_err = np.std(trf_ori_all, axis=0)/np.sqrt(subject_num)
trf_chimera_all_ave = np.average(trf_chimera_all, axis=0)
trf_chimera_all_err = np.std(trf_chimera_all, axis=0)/np.sqrt(subject_num)

# Plot
chn=1
plt.figure()
# Original
plt.plot(lag, trf_ori_all_ave[0,:,chn], c='gray', linewidth=2, linestyle='solid', label="Original-Note onset")
plt.fill_between(lag, trf_ori_all_ave[0,:,chn]-trf_ori_all_err[0,:,chn], trf_ori_all_ave[0,:,chn]+trf_ori_all_err[0,:,chn], color='gray', alpha=0.8)
plt.plot(lag, trf_ori_all_ave[1,:,chn], c='C0', linewidth=2, linestyle='solid', label="Original-Downbeat")
plt.fill_between(lag, trf_ori_all_ave[1,:,chn]-trf_ori_all_err[1,:,chn], trf_ori_all_ave[1,:,chn]+trf_ori_all_err[1,:,chn], color='C0',alpha=0.5)
#Chimera
plt.plot(lag, trf_chimera_all_ave[0,:,chn], c='darkgray', linewidth=1, linestyle='dashed', label="Chimera-Note onset")
plt.fill_between(lag, trf_chimera_all_ave[0,:,chn]-trf_chimera_all_err[0,:,chn], trf_chimera_all_ave[0,:,chn]+trf_chimera_all_err[0,:,chn], color='darkgray', alpha=0.8)
plt.plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch")
plt.fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Downbeat timing")
plt.fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
plt.plot(lag, trf_chimera_all_ave[3,:,chn], c='C3', linewidth=1, linestyle='dashed', label="Chimera-Downbeat pitch*timing")
plt.fill_between(lag, trf_chimera_all_ave[3,:,chn]-trf_chimera_all_err[3,:,chn], trf_chimera_all_ave[3,:,chn]+trf_chimera_all_err[3,:,chn], color='C3',alpha=0.5)
#plt.fill_between(lag, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
#plt.ylim(-2e-5, 4e-5)
plt.grid()
plt.title("Group 4 \n channel: " + channel_names[chn])
plt.legend()
plt.ylim(-0.0002, 0.0005)
plt.show()

# plot stack individual
# Original downbeat
fig = plt.figure()
fig.set_size_inches(3.5, 8)
for i in subject_index_sort:
    plt.plot(lag, trf_ori_all[i,1,:,chn] + i*5e-5, c='C0',alpha=0.4+0.02*i)
    #plt.hlines(i*5e-5, -0.1, 0.7, color='k', alpha=0.6)
plt.vlines(0.19, 0, 0.00135, color='k', alpha=0.6)
#plt.grid()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['left'].set_visible(False)
plt.tight_layout()

# %% Computing amplitude-training correlation
from statsmodels.stats import multitest
subject_index = np.arange(0,27,1)
music_training_years = [8,23,1,5,12,0,3,1,2,20,5,3.5,7,7,0,0,1,11,10,10,0,8,14,14,10,14,6]
formal_training_years = [4,10,1.5,5,1.5,0,3,1,2,10,5,2,7,7,0,0,1,11,5,10,0,2,10,5,2,6,2]


multitest.fdrcorrection([0.02051007621967423,0.00984645790336051,0.07100147412864843,0.055563507133907,0.07100147412864843,0.1084397410392292],method='i')
_, p_corr = multitest.fdrcorrection([0.0028777016899068975, 0.0718420174655359, 0.025101145652633574, 0.9166512919771622,0.5270080689141154],method='i')

chn=1
# Original onset amplitude

# P1 
ori_onset_amplitude_p1 = []
ind = np.where(ori_onset_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_ori_all[si,0,i,chn] for i in np.arange(16,29)])
    ori_onset_amplitude_p1 += [y]
    
m, b = np.polyfit(formal_training_years, ori_onset_amplitude_p1, 1)
#stats.pearsonr(music_training_years, ori_onset_amplitude)
r, p = stats.pearsonr(formal_training_years, ori_onset_amplitude_p1)

plt.figure()
plt.scatter(formal_training_years, ori_onset_amplitude_p1)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Formal music training years")
plt.title("Original Note onset TRF P1 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p))
plt.tight_layout()
plt.grid()

# P2
ori_onset_amplitude_p2 = []
ind = np.where(ori_onset_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_ori_all[si,0,i,chn] for i in np.arange(28,47)])
    ori_onset_amplitude_p2 += [y]
    
m, b = np.polyfit(formal_training_years, ori_onset_amplitude_p2, 1)
#stats.pearsonr(music_training_years, ori_onset_amplitude)
r, p = stats.pearsonr(formal_training_years, ori_onset_amplitude_p2)

fig = plt.figure(dpi=dpi)
fig.set_size_inches(4, 4)
plt.scatter(formal_training_years, ori_onset_amplitude_p2)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Formal music training years")
plt.title("Original Note onset TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p))
plt.tight_layout()
plt.grid()
plt.savefig(figure_path+'ABR_results.png', dpi=dpi, format='png')


# Original downbeat amplitude
ori_db_amplitude = []
ind = np.where(ori_downbeat_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_ori_all[si,1,i,chn] for i in ind])
    ori_db_amplitude += [y]
    
m, b = np.polyfit(formal_training_years, ori_db_amplitude, 1)
stats.pearsonr(music_training_years, ori_db_amplitude)
r,p=stats.pearsonr(formal_training_years, ori_db_amplitude)

plt.figure()
plt.scatter(formal_training_years, ori_db_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Formal music training years")
#plt.title("Original Downbeat TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[1]))
plt.grid()

# Chimera onset
# P1
chimera_onset_amplitude_p1 = []
ind = np.where(chimera_onset_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_chimera_all[si,0,i,chn] for i in np.arange(16,29)])
    chimera_onset_amplitude_p1 += [y]
    
m, b = np.polyfit(formal_training_years, chimera_onset_amplitude_p1, 1)
stats.pearsonr(music_training_years, chimera_onset_amplitude_p1)
r,p=stats.pearsonr(formal_training_years, chimera_onset_amplitude_p1)

plt.figure()
plt.scatter(formal_training_years, chimera_onset_amplitude_p1)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Note onset TRF P1 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p))
plt.grid()

# p2
chimera_pitch_amplitude_2 = []
ind = np.where(chimera_pitch_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_chimera_all[si,0,i,chn] for i in np.arange(28,47)])
    chimera_pitch_amplitude_2 += [y]
    
m, b = np.polyfit(formal_training_years, chimera_pitch_amplitude_2, 1)
#stats.pearsonr(music_training_years, chimera_pitch_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_pitch_amplitude_2)

plt.figure()
plt.scatter(formal_training_years, chimera_pitch_amplitude_2)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Note onset TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p))
plt.grid()

# Chimera timing downbeat
chimera_timing_amplitude = []
ind = np.where(chimera_timing_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_chimera_all[si,2,i,chn] for i in ind])
    chimera_timing_amplitude += [y]
    
m, b = np.polyfit(formal_training_years, chimera_timing_amplitude, 1)
stats.pearsonr(music_training_years, chimera_timing_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_timing_amplitude)

plt.figure()
plt.scatter(formal_training_years, chimera_timing_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Timing Downbeat TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[4]))
plt.grid()

# Chimera both downbeat
chimera_db_amplitude = []
ind = np.where(chimera_inter_significant_points[1,:]==True)[0]
for si in subject_index:
    y = max([trf_chimera_all[si,3,i,chn] for i in ind])
    chimera_db_amplitude += [y]
    
m, b = np.polyfit(formal_training_years, chimera_db_amplitude, 1)
stats.pearsonr(music_training_years, chimera_db_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_db_amplitude)

plt.figure()
plt.scatter(formal_training_years, chimera_db_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Pitch*Timing TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[5]))
plt.grid()

# %% All channel correlation
subject_index = np.arange(0,27,1)
formal_training_years = [4,10,1.5,5,1.5,0,3,1,2,10,5,2,7,7,0,0,1,11,5,10,0,2,10,5,2,6,2]
_, p_corr = multitest.fdrcorrection([0.0028777016899068975, 0.0718420174655359, 0.025101145652633574, 0.9166512919771622,0.5270080689141154],method='i')

# Original onset amplitude
# P1
ori_onset_amplitude_p1_all = []
r_val = []
p_val = []
for chn in range(len(channel_names)):
    ori_onset_amplitude_p1 = []
    for si in subject_index:
        y = max([trf_ori_all[si,0,i,chn] for i in np.arange(16,29)])
        ori_onset_amplitude_p1 += [y]
    ori_onset_amplitude_p1_all += [ori_onset_amplitude_p1]
    r, p = stats.pearsonr(formal_training_years, ori_onset_amplitude_p1)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')
    
chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Original Note onset P1", fontsize=15)
plt.savefig(figure_path+"Original_onset_P1amp_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

# P2
ori_onset_amplitude_p2_all = []
r_val = []
p_val = []

for chn in range(len(channel_names)):
    ori_onset_amplitude_p2 = []
    for si in subject_index:
        y = max([trf_ori_all[si,0,i,chn] for i in np.arange(28,47)])
        ori_onset_amplitude_p2 += [y]
    ori_onset_amplitude_p2_all += [ori_onset_amplitude_p2]
    r, p = stats.pearsonr(formal_training_years, ori_onset_amplitude_p2)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')

chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Original Note onset P2", fontsize=15)
plt.savefig(figure_path+"Original_onset_P2amp_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

# Correlation with fitted line
m, b = np.polyfit(formal_training_years, ori_onset_amplitude_p2_all[1], 1)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6,4)
plt.scatter(formal_training_years, ori_onset_amplitude_p2_all[1])
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel("Amplitude (AU)", fontsize=15)
plt.xlabel("Formal music training years",fontsize=15)
plt.title("Original Note onset P2 (Fz)", fontsize=15)
plt.grid()
plt.show()
plt.savefig(figure_path+"Original_onset_P2amp_musicianship_corr_scatter.png",dpi=dpi,format='png', bbox_inches='tight')



# Original downbeat amplitude
r_val = []
p_val = []
ori_db_amplitude_all = []
for chn in range(len(channel_names)):
    ori_db_amplitude = []
    for si in subject_index:
        y = max([trf_ori_all[si,1,i,chn] for i in np.arange(25,64)])
        ori_db_amplitude += [y]
    ori_db_amplitude_all += [ori_db_amplitude]
    r, p = stats.pearsonr(formal_training_years, ori_db_amplitude)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')

chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Original Downbeat peak", fontsize=15)
plt.savefig(figure_path+"Original_downbeat_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

m, b = np.polyfit(formal_training_years, ori_db_amplitude, 1)

plt.figure()
plt.scatter(formal_training_years, ori_db_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Formal music training years")
#plt.title("Original Downbeat TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[1]))
plt.grid()

# Chimera onset
# P1
chimera_onset_amplitude_p1_all = []
r_val = []
p_val = []
for chn in range(len(channel_names)):
    chimera_onset_amplitude_p1 = []
    for si in subject_index:
        y = max([trf_chimera_all[si,0,i,chn] for i in np.arange(16,29)])
        chimera_onset_amplitude_p1 += [y]
    chimera_onset_amplitude_p1_all += [chimera_onset_amplitude_p1]
    r, p = stats.pearsonr(formal_training_years, chimera_onset_amplitude_p1)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')
    
chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Chimera Note onset P1", fontsize=15)
plt.savefig(figure_path+"chimera_onset_P1amp_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

# P2
chimera_onset_amplitude_p2_all = []
r_val = []
p_val = []

for chn in range(len(channel_names)):
    chimera_onset_amplitude_p2 = []
    for si in subject_index:
        y = max([trf_chimera_all[si,0,i,chn] for i in np.arange(28,47)])
        chimera_onset_amplitude_p2 += [y]
    chimera_onset_amplitude_p2_all += [chimera_onset_amplitude_p2]
    r, p = stats.pearsonr(formal_training_years, chimera_onset_amplitude_p2)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')

chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Chimera Note onset P2", fontsize=15)
plt.savefig(figure_path+"chimera_onset_P2amp_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

m, b = np.polyfit(formal_training_years, chimera_pitch_amplitude_2, 1)
#stats.pearsonr(music_training_years, chimera_pitch_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_pitch_amplitude_2)

plt.figure()
plt.scatter(formal_training_years, chimera_pitch_amplitude_2)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Note onset TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p))
plt.grid()

# Chimera timing downbeat
chimera_timing_amplitude_all = []
r_val = []
p_val = []

for chn in range(len(channel_names)):
    chimera_timing_amplitude = []
    for si in subject_index:
        y = max([trf_chimera_all[si,2,i,chn] for i in  np.arange(28,47)])
        chimera_timing_amplitude += [y]
    chimera_timing_amplitude_all += [chimera_timing_amplitude]
    r, p = stats.pearsonr(formal_training_years, chimera_timing_amplitude)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')

chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
fig.suptitle("Chimera Timing Peak", fontsize=15)
plt.savefig(figure_path+"chimera_timing_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')

m, b = np.polyfit(formal_training_years, chimera_timing_amplitude, 1)
stats.pearsonr(music_training_years, chimera_timing_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_timing_amplitude)

plt.figure()
plt.scatter(formal_training_years, chimera_timing_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Timing Downbeat TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[4]))
plt.grid()

# Chimera interaction downbeat
chimera_timing_amplitude_all = []
r_val = []
p_val = []

for chn in range(len(channel_names)):
    chimera_timing_amplitude = []
    for si in subject_index:
        y = max([trf_chimera_all[si,3,i,chn] for i in  np.arange(28,47)])
        chimera_timing_amplitude += [y]
    chimera_timing_amplitude_all += [chimera_timing_amplitude]
    r, p = stats.pearsonr(formal_training_years, chimera_timing_amplitude)
    r_val += [r]
    p_val += [p]
_, p_corr = multitest.fdrcorrection(p_val, method='i')

chn_sig = p_corr <0.05

# Plot topomap of R values
fig,ax1 = plt.subplots(1)
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
im, cm = mne.viz.plot_topomap(np.array(r_val), pos=info, axes=ax1, vlim=(-0.6, 0.6), 
                              image_interp="nearest", mask=chn_sig, contours=0,
                              mask_params=mask_params)
# manually fiddle the position of colorbar
ax_x_start = 0.90
ax_x_width = 0.04
ax_y_start = 0.1
ax_y_height = 0.7
cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
clb = fig.colorbar(im, cax=cbar_ax)
clb.ax.set_title("Pearson's R",fontsize=10)
fig.suptitle("Chimera Interaction Peak", fontsize=15)
plt.savefig(figure_path+"chimera_inter_musicianship_corr.png",dpi=dpi,format='png', bbox_inches='tight')
    
m, b = np.polyfit(formal_training_years, chimera_db_amplitude, 1)
stats.pearsonr(music_training_years, chimera_db_amplitude)
r,p=stats.pearsonr(formal_training_years, chimera_db_amplitude)

plt.figure()
plt.scatter(formal_training_years, chimera_db_amplitude)
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ylabel("Amplitude")
plt.xlabel("Music training years")
plt.title("Chimera Pitch*Timing TRF P2 amplitude \n Pearson's r={:.3f} \n p_corr={:.3f}".format(r,p_corr[5]))
plt.grid()

