#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:28:27 2024

@author: tshan
"""

import numpy as np
import pandas as pd
import scipy.signal as signal
import scipy.stats as stats
from mne.filter import resample
from expyfun.io import read_wav, write_hdf5, read_hdf5
from mtrf.model import TRF, load_sample_data
#from mtrf.stats import cross_validate
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
exp_path = '/media/tshan@urmc-sh.rochester.edu/TOSHIBA/Chimera/'
exp_path = 'F://Chimera/'


# %%
dpi = 300
data_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/'
data_path = 'D://AMPLab/MusicABR/diverse_dataset/music_abr_diverse_beh/'

figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/poster/'
figure_path = 'D://AMPLab/MusicExp/paper_figs/'

plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

# %% Average data
from scipy.stats import ttest_ind
import mne
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test

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



trf_ori_AM = np.zeros((len(subject_list),10,102,32))
trf_ori_LTM = np.zeros((len(subject_list),6,102,32))
trf_ori_STM = np.zeros((len(subject_list),6,102,32))
trf_ori_both = np.zeros((len(subject_list),6,102,32))
trf_ori_A = np.zeros((len(subject_list),2,102,32))
trf_chimera_AM = np.zeros((len(subject_list),10,102,32))
trf_chimera_LTM = np.zeros((len(subject_list),6,102,32))
trf_chimera_STM = np.zeros((len(subject_list),6,102,32))
trf_chimera_both = np.zeros((len(subject_list),6,102,32))
trf_chimera_A = np.zeros((len(subject_list),2,102,32))
# pair_diff_ori = np.zeros((len(subject_list),33))
# pair_diff_chimera = np.zeros((len(subject_list),33))
#data_matrix = np.zeros((len(subject_list),2,6,151,32))
ori_AM_correlation = np.zeros((len(subject_list),66))
ori_LTM_correlation = np.zeros((len(subject_list),66))
ori_STM_correlation = np.zeros((len(subject_list),66))
ori_both_correlation = np.zeros((len(subject_list),66))
ori_A_correlation = np.zeros((len(subject_list),66))
chimera_AM_correlation = np.zeros((len(subject_list),66))
chimera_LTM_correlation = np.zeros((len(subject_list),66))
chimera_STM_correlation = np.zeros((len(subject_list),66))
chimera_both_correlation = np.zeros((len(subject_list),66))
chimera_A_correlation = np.zeros((len(subject_list),66))
set_num_ori = np.zeros((len(subject_list),66))
set_num_chimera = np.zeros((len(subject_list),66))
for i, subject in enumerate(subject_list):
    data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
    trf_data = read_hdf5(data_root+'/'+subject+'_32chn_eeg_125hz_ltm_stm_both_nb_pitch_ioi_hp1_lp8_ICA_trf_data_-02_1_corr_all_crossval_best_reg_individual.hdf5')
    trf_ori_AM[i] = trf_data['ori_AM_weights']
    trf_ori_LTM[i] = trf_data['ori_LTM_weights']
    trf_ori_STM[i] = trf_data['ori_STM_weights']
    trf_ori_both[i] = trf_data['ori_both_weights']
    trf_ori_A[i] =trf_data['ori_A_weights']
    #data_matrix[i,0,:,:,:] = trf_data['ori_AM_weights']
    trf_chimera_AM[i] = trf_data['chimera_AM_weights']
    trf_chimera_LTM[i] = trf_data['chimera_LTM_weights']
    trf_chimera_STM[i] = trf_data['chimera_STM_weights']
    trf_chimera_both[i] = trf_data['chimera_both_weights']
    trf_chimera_A[i] = trf_data['chimera_A_weights']
    ori_AM_correlation[i] = trf_data['ori_AM_correlation']
    ori_LTM_correlation[i] = trf_data['ori_LTM_correlation']
    ori_STM_correlation[i] = trf_data['ori_STM_correlation']
    ori_both_correlation[i] = trf_data['ori_both_correlation']
    ori_A_correlation[i] = trf_data['ori_A_correlation']
    chimera_AM_correlation[i] = trf_data['chimera_AM_correlation']
    chimera_LTM_correlation[i] = trf_data['chimera_LTM_correlation']
    chimera_STM_correlation[i] = trf_data['chimera_STM_correlation']
    chimera_both_correlation[i] = trf_data['chimera_both_correlation']
    chimera_A_correlation[i] = trf_data['chimera_A_correlation']
    set_num_ori[i] = trf_data['set_num_ori']
    set_num_chimera[i] = trf_data['set_num_chimera']
    time = trf_data["time"]
    #pair_diff_ori[i] = trf_data['pairs_ori_diff']
    #pair_diff_chimera[i] = trf_data['pairs_chimera_diff']
    #data_matrix[i,1,:,:,:] = trf_data['chimera_AM_weights']

trf_ori_AM_ave = np.mean(trf_ori_both, axis=0)
trf_ori_AM_se = np.std(trf_ori_both, axis=0)/np.sqrt(len(subject_list))
trf_chimera_AM_ave = np.mean(trf_chimera_both, axis=0)
trf_chimera_AM_se = np.std(trf_chimera_both, axis=0)/np.sqrt(len(subject_list))

# %% Weight Stats
def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)
p_raw = np.zeros((len(channel_names),6,151))
sig_time = np.zeros((len(channel_names),6,151))
p_corrected = np.zeros((len(channel_names),6,151))
sig_time_corrected = np.zeros((len(channel_names),6,151))
alpha=0.05
for chi in range(32):
    for reg in range(6):
        res = stats.permutation_test((trf_ori_AM[:,reg,:,chi], trf_chimera_AM[:,reg,:,chi]),statistic,permutation_type='samples',axis=0)
        p_raw[chi,reg,:] = res.pvalue
        indices = np.where(p_raw[chi,reg,:]<=alpha)[0]
        sig_time[chi,reg,indices] = 1
        sig_time_corrected[chi,reg,:], p_corrected[chi,reg,:] = multitest.fdrcorrection(res.pvalue,alpha=alpha)

        #w, p_raw[chi,reg,:] = stats.wilcoxon(trf_ori_AM[:,reg,:,chi], trf_chimera_AM[:,reg,:,chi],axis=0)
        #sig_time[chi,reg,:], p_corrected[chi,reg,:] = multitest.fdrcorrection(p_raw[chi,reg,44:69],alpha=0.1)

# %% Clustering plots
from scipy.stats import ttest_ind
import mne
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test

montage = mne.channels.make_standard_montage("easycap-M1")
info = mne.create_info(channel_names, sfreq=eeg_fs_down,ch_types='eeg')
info.set_montage(montage)

###### Original vs Chimera Flux 
trf_ori_AM_flux = trf_ori_AM[:,0,:,:]
trf_ori_AM_flux = np.swapaxes(trf_ori_AM_flux, 1, 2)
trf_chimera_AM_flux = trf_chimera_AM[:,0,:,:]
trf_chimera_AM_flux = np.swapaxes(trf_chimera_AM_flux, 1, 2)
Epochs_ori_AM_flux = mne.EpochsArray(trf_ori_AM_flux, info, tmin=-0.2)
Epochs_ori_AM_flux.set_montage(montage)
Epochs_chimera_AM_flux = mne.EpochsArray(trf_chimera_AM_flux, info, tmin=-0.2)
Epochs_chimera_AM_flux.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_ori_AM_flux.info, "eeg")

X = [Epochs_ori_AM_flux.get_data().transpose(0,2,1),
     Epochs_chimera_AM_flux.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_flux.average(), Epochs_chimera_AM_flux.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera Flux", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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

###### Original vs Chimera onset 
trf_ori_AM_onset = trf_ori_AM[:,1,:,:]
trf_ori_AM_onset = np.swapaxes(trf_ori_AM_onset, 1, 2)
trf_chimera_AM_onset = trf_chimera_AM[:,1,:,:]
trf_chimera_AM_onset = np.swapaxes(trf_chimera_AM_onset, 1, 2)
Epochs_ori_AM_onset = mne.EpochsArray(trf_ori_AM_onset, info, tmin=-0.2)
Epochs_ori_AM_onset.set_montage(montage)
Epochs_chimera_AM_onset = mne.EpochsArray(trf_chimera_AM_onset, info, tmin=-0.2)
Epochs_chimera_AM_onset.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_onset.info, "eeg")

X = [Epochs_ori_AM_onset.get_data().transpose(0,2,1),
     Epochs_chimera_AM_onset.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_onset.average(), Epochs_chimera_AM_onset.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera onset", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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



###### Original vs Chimera pitch surprisal 
trf_ori_AM_sp = trf_ori_AM[:,2,:,:]
trf_ori_AM_sp = np.swapaxes(trf_ori_AM_sp, 1, 2)
trf_chimera_AM_sp = trf_chimera_AM[:,2,:,:]
trf_chimera_AM_sp = np.swapaxes(trf_chimera_AM_sp, 1, 2)
Epochs_ori_AM_sp = mne.EpochsArray(trf_ori_AM_sp, info, tmin=-0.2)
Epochs_ori_AM_sp.set_montage(montage)
Epochs_chimera_AM_sp = mne.EpochsArray(trf_chimera_AM_sp, info, tmin=-0.2)
Epochs_chimera_AM_sp.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_sp.info, "eeg")

X = [Epochs_ori_AM_sp.get_data().transpose(0,2,1),
     Epochs_chimera_AM_sp.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_sp.average(), Epochs_chimera_AM_sp.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera pitch surprisal", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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


###### Original vs Chimera pitch entropy 
trf_ori_AM_hp = trf_ori_AM[:,3,:,:]
trf_ori_AM_hp = np.swapaxes(trf_ori_AM_hp, 1, 2)
trf_chimera_AM_hp = trf_chimera_AM[:,3,:,:]
trf_chimera_AM_hp = np.swapaxes(trf_chimera_AM_hp, 1, 2)
Epochs_ori_AM_hp = mne.EpochsArray(trf_ori_AM_hp, info, tmin=-0.2)
Epochs_ori_AM_hp.set_montage(montage)
Epochs_chimera_AM_hp = mne.EpochsArray(trf_chimera_AM_hp, info, tmin=-0.2)
Epochs_chimera_AM_hp.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_hp.info, "eeg")

X = [Epochs_ori_AM_hp.get_data().transpose(0,2,1),
     Epochs_chimera_AM_hp.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_hp.average(), Epochs_chimera_AM_hp.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera Pitch Entropy", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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


###### Original vs Chimera onset surprisal 
trf_ori_AM_so = trf_ori_AM[:,4,:,:]
trf_ori_AM_so = np.swapaxes(trf_ori_AM_so, 1, 2)
trf_chimera_AM_so = trf_chimera_AM[:,4,:,:]
trf_chimera_AM_so = np.swapaxes(trf_chimera_AM_so, 1, 2)
Epochs_ori_AM_so = mne.EpochsArray(trf_ori_AM_so, info, tmin=-0.2)
Epochs_ori_AM_so.set_montage(montage)
Epochs_chimera_AM_so = mne.EpochsArray(trf_chimera_AM_so, info, tmin=-0.2)
Epochs_chimera_AM_so.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_hp.info, "eeg")

X = [Epochs_ori_AM_so.get_data().transpose(0,2,1),
     Epochs_chimera_AM_so.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_so.average(), Epochs_chimera_AM_so.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera Timing Surprisal", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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


###### Original vs Chimera onset entropy 
trf_ori_AM_ho = trf_ori_AM[:,5,:,:]
trf_ori_AM_ho = np.swapaxes(trf_ori_AM_ho, 1, 2)
trf_chimera_AM_ho = trf_chimera_AM[:,5,:,:]
trf_chimera_AM_ho = np.swapaxes(trf_chimera_AM_ho, 1, 2)
Epochs_ori_AM_ho = mne.EpochsArray(trf_ori_AM_ho, info, tmin=-0.2)
Epochs_ori_AM_ho.set_montage(montage)
Epochs_chimera_AM_ho = mne.EpochsArray(trf_chimera_AM_ho, info, tmin=-0.2)
Epochs_chimera_AM_ho.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_ho.info, "eeg")

X = [Epochs_ori_AM_ho.get_data().transpose(0,2,1),
     Epochs_chimera_AM_ho.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_ho.average(), Epochs_chimera_AM_ho.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera Timing Entopy", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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

###### Original vs Chimera STM St
trf_ori_AM_so = trf_ori_AM[:,8,:,:]
trf_ori_AM_so = np.swapaxes(trf_ori_AM_so, 1, 2)
trf_chimera_AM_so = trf_chimera_AM[:,8,:,:]
trf_chimera_AM_so = np.swapaxes(trf_chimera_AM_so, 1, 2)
Epochs_ori_AM_so = mne.EpochsArray(trf_ori_AM_so, info, tmin=-0.2)
Epochs_ori_AM_so.set_montage(montage)
Epochs_chimera_AM_so = mne.EpochsArray(trf_chimera_AM_so, info, tmin=-0.2)
Epochs_chimera_AM_so.set_montage(montage)

adjacency, _ = find_ch_adjacency(Epochs_chimera_AM_so.info, "eeg")

X = [Epochs_ori_AM_so.get_data().transpose(0,2,1),
     Epochs_chimera_AM_so.get_data().transpose(0,2,1)]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked = mne.combine_evoked([Epochs_ori_AM_so.average(), Epochs_chimera_AM_so.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked.plot_joint(
    title="Original vs Chimera Timing Surprisal", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8, 8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked.plot_image(
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

# %% TRF traces plots
regressors = ['spectral flux','note onset','LTM-Sp','LTM-Ep',
              'LTM-St','LTM-Et','STM-Sp',
              'STM-Ep','STM-St','STM-Et']
regressors = ['spectral flux','note onset','Sp','Ep',
              'St','Et']
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

reg=0
chi_n=1

for reg in range(len(regressors)):
#    for chi_n in range(32):
    for chi_n in [1]:
        fig = plt.figure()
        fig.set_size_inches(3,3)
        plt.plot(time*1e3, trf_ori_AM_ave[reg,:,chi_n],c=colors[reg], linewidth=2, linestyle='solid', label="Original")
        plt.fill_between(time*1e3, trf_ori_AM_ave[reg,:,chi_n]-trf_ori_AM_se[reg,:,chi_n], trf_ori_AM_ave[reg,:,chi_n]+trf_ori_AM_se[reg,:,chi_n], color=colors[reg], alpha=0.8)
        plt.plot(time*1e3, trf_chimera_AM_ave[reg,:,chi_n], c=colors[reg], linewidth=1, linestyle='dashed', label="Chimeric")
        plt.fill_between(time*1e3, trf_chimera_AM_ave[reg,:,chi_n]-trf_chimera_AM_se[reg,:,chi_n], trf_chimera_AM_ave[reg,:,chi_n]+trf_chimera_AM_se[reg,:,chi_n], color=colors[reg],alpha=0.5)
        #plt.fill_between(time, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
        plt.xlim(-100,700)
        plt.ylim(-2e-5, 4e-5)
        plt.grid()
        plt.title(regressors[reg])
        plt.legend(fontsize=10)
        plt.ylabel("Magnitude (AU)")
        plt.xlabel("Time (ms)")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
        plt.savefig(figure_path+'zscore_reg_individual_'+regressors[reg]+'_'+channel_names[chi_n]+'.svg', dpi=dpi, format='svg')

plt.figure()
chi_n=1
plt.plot(time, (trf_ori_AM_ave[0,:,chi_n]-trf_chimera_AM_ave[0,:,chi_n]),c="C0", linewidth=2)
plt.plot(time, (trf_ori_AM_ave[1,:,chi_n]-trf_chimera_AM_ave[1,:,chi_n]),c="C1", linewidth=2)
plt.plot(time, (trf_ori_AM_ave[2,:,chi_n]-trf_chimera_AM_ave[2,:,chi_n]),c="C2", linewidth=2)
plt.plot(time, (trf_ori_AM_ave[3,:,chi_n]-trf_chimera_AM_ave[3,:,chi_n]),c="C3", linewidth=2)
plt.plot(time, (trf_ori_AM_ave[4,:,chi_n]-trf_chimera_AM_ave[4,:,chi_n]),c="C4", linewidth=2)
plt.plot(time, (trf_ori_AM_ave[5,:,chi_n]-trf_chimera_AM_ave[5,:,chi_n]),c="C5", linewidth=2)

plt.legend(['flux','onset','Sp','Hp','So','Ho'])
plt.grid()
plt.show()

#### Topographs
montage = mne.channels.make_standard_montage("standard_1020")
info = mne.create_info(channel_names, sfreq=eeg_fs_down, ch_types='eeg')

# ind
ori_evoked_flux = mne.EvokedArray(trf_ori_AM[11,1,:,:].T,info, tmin=-0.15, kind="average")
ori_evoked_flux.set_montage(montage)
ori_evoked_flux.nave = len(subject_list)
ori_evoked_flux.plot_joint(times = [50e-3, 140e-3, 250e-3, 350e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Flux")
chimera_evoked_flux = mne.EvokedArray(trf_chimera_AM[11,0,:,:].T,info, tmin=-0.15, kind="average")
chimera_evoked_flux.set_montage(montage)
chimera_evoked_flux.nave = len(subject_list)
chimera_evoked_flux.plot_joint(times = [50e-3, 140e-3, 250e-3, 350e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="chimera_evoked_flux - Flux")

# Flux
ori_evoked_flux_ave = mne.EvokedArray(trf_ori_AM_ave[0].T,info, tmin=-0.15, kind="average")
ori_evoked_flux_se = mne.EvokedArray(trf_ori_AM_se[0].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_flux_ave.set_montage(montage)
ori_evoked_flux_ave.nave = len(subject_list)
ori_evoked_flux_ave.plot_joint(times = [100e-3, 200e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Flux")

chimera_evoked_flux_ave = mne.EvokedArray(trf_chimera_AM_ave[0].T,info, tmin=-0.15, kind="average")
chimera_evoked_flux_se = mne.EvokedArray(trf_chimera_AM_se[0].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_flux_ave.set_montage(montage)
chimera_evoked_flux_ave.nave = len(subject_list)
chimera_evoked_flux_ave.plot_joint(times = [100e-3, 200e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Flux")

# Onset
ori_evoked_onset_ave = mne.EvokedArray(trf_ori_AM_ave[1].T,info, tmin=-0.15, kind="average")
ori_evoked_onset_se = mne.EvokedArray(trf_ori_AM_se[1].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_onset_ave.set_montage(montage)
ori_evoked_onset_ave.nave = len(subject_list)
ori_evoked_onset_ave.plot_joint(times = [100e-3, 200e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - onset")

chimera_evoked_onset_ave = mne.EvokedArray(trf_chimera_AM_ave[1].T,info, tmin=-0.15, kind="average")
chimera_evoked_onset_se = mne.EvokedArray(trf_chimera_AM_se[1].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_onset_ave.set_montage(montage)
chimera_evoked_onset_ave.nave = len(subject_list)
chimera_evoked_onset_ave.plot_joint(times = [100e-3, 200e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - onset")

# Suprisal pitch
ori_evoked_sp_ave = mne.EvokedArray(trf_ori_AM_ave[2].T,info, tmin=-0.15, kind="average")
ori_evoked_sp_se = mne.EvokedArray(trf_ori_AM_se[2].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_sp_ave.set_montage(montage)
ori_evoked_sp_ave.nave = len(subject_list)
ori_evoked_sp_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Suprisal pitch")

chimera_evoked_sp_ave = mne.EvokedArray(trf_chimera_AM_ave[2].T,info, tmin=-0.15, kind="average")
chimera_evoked_sp_se = mne.EvokedArray(trf_chimera_AM_se[2].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_sp_ave.set_montage(montage)
chimera_evoked_sp_ave.nave = len(subject_list)
chimera_evoked_sp_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Suprisal pitch")

# Entropy pitch
ori_evoked_hp_ave = mne.EvokedArray(trf_ori_AM_ave[3].T,info, tmin=-0.15, kind="average")
ori_evoked_hp_se = mne.EvokedArray(trf_ori_AM_se[3].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_hp_ave.set_montage(montage)
ori_evoked_hp_ave.nave = len(subject_list)
ori_evoked_hp_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Entropy pitch")

chimera_evoked_hp_ave = mne.EvokedArray(trf_chimera_AM_ave[3].T,info, tmin=-0.15, kind="average")
chimera_evoked_hp_se = mne.EvokedArray(trf_chimera_AM_se[3].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_hp_ave.set_montage(montage)
chimera_evoked_hp_ave.nave = len(subject_list)
chimera_evoked_hp_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Entropy pitch")

# Suprisal onset
ori_evoked_so_ave = mne.EvokedArray(trf_ori_AM_ave[4].T,info, tmin=-0.15, kind="average")
ori_evoked_so_se = mne.EvokedArray(trf_ori_AM_se[4].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_so_ave.set_montage(montage)
ori_evoked_so_ave.nave = len(subject_list)
ori_evoked_so_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Suprisal onset")

chimera_evoked_so_ave = mne.EvokedArray(trf_chimera_AM_ave[4].T,info, tmin=-0.15, kind="average")
chimera_evoked_so_se = mne.EvokedArray(trf_chimera_AM_se[4].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_so_ave.set_montage(montage)
chimera_evoked_so_ave.nave = len(subject_list)
chimera_evoked_so_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Suprisal onset")

# Entropy onset
ori_evoked_ho_ave = mne.EvokedArray(trf_ori_AM_ave[5].T,info, tmin=-0.15, kind="average")
ori_evoked_ho_se = mne.EvokedArray(trf_ori_AM_se[5].T,info, tmin=-0.15, kind="standard_error")
ori_evoked_ho_ave.set_montage(montage)
ori_evoked_ho_ave.nave = len(subject_list)
ori_evoked_ho_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Original - Entropy onset")

chimera_evoked_ho_ave = mne.EvokedArray(trf_chimera_AM_ave[5].T,info, tmin=-0.15, kind="average")
chimera_evoked_ho_se = mne.EvokedArray(trf_chimera_AM_se[5].T,info, tmin=-0.15, kind="standard_error")
chimera_evoked_ho_ave.set_montage(montage)
chimera_evoked_ho_ave.nave = len(subject_list)
chimera_evoked_ho_ave.plot_joint(times = [100e-3, 200e-3, 250e-3, 300e-3, 400e-3, 500e-3],
                            ts_args=dict(ylim=dict(eeg=[-3e-5, 5e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-3e-5, 5e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Chimera - Entropy onset")

##### Difference
trf_diff_ave = trf_ori_AM_ave - trf_chimera_AM_ave
# flux
diff_evoked_flux_ave = mne.EvokedArray(trf_diff_ave[0].T,info, tmin=-0.15, kind="average")
diff_evoked_flux_ave.set_montage(montage)
diff_evoked_flux_ave.nave = len(subject_list)
diff_evoked_flux_ave.plot_joint(times = [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Flux")

# onset
diff_evoked_onset_ave = mne.EvokedArray(trf_diff_ave[1].T,info, tmin=-0.15, kind="average")
diff_evoked_onset_ave.set_montage(montage)
diff_evoked_onset_ave.nave = len(subject_list)
diff_evoked_onset_ave.plot_joint(times = [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Onset")

# Surprisal pitch
diff_evoked_sp_ave = mne.EvokedArray(trf_diff_ave[2].T,info, tmin=-0.15, kind="average")
diff_evoked_sp_ave.set_montage(montage)
diff_evoked_sp_ave.nave = len(subject_list)
diff_evoked_sp_ave.plot_joint(times = [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Surprisal Pitch")

# Entropy pitch
diff_evoked_hp_ave = mne.EvokedArray(trf_diff_ave[3].T,info, tmin=-0.15, kind="average")
diff_evoked_hp_ave.set_montage(montage)
diff_evoked_hp_ave.nave = len(subject_list)
diff_evoked_hp_ave.plot_joint(times =  [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Entropy Pitch")

# Surprisal onset
diff_evoked_so_ave = mne.EvokedArray(trf_diff_ave[4].T,info, tmin=-0.15, kind="average")
diff_evoked_so_ave.set_montage(montage)
diff_evoked_so_ave.nave = len(subject_list)
diff_evoked_so_ave.plot_joint(times = [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Surprisal Onset")

# Entropy onset
diff_evoked_ho_ave = mne.EvokedArray(trf_diff_ave[5].T,info, tmin=-0.15, kind="average")
diff_evoked_ho_ave.set_montage(montage)
diff_evoked_ho_ave.nave = len(subject_list)
diff_evoked_ho_ave.plot_joint(times = [50e-3,150e-3,200e-3,250e-3,300e-3,400e-3,450e-3,550e-3,600e-3,700e-3],
                            ts_args=dict(ylim=dict(eeg=[-1e-5, 1e-5]), xlim=[-150, 750],
                                         units='TRF (AU)', time_unit='ms', gfp=True,
                                         scalings=dict(eeg=1)),
                            topomap_args=dict(vlim=(-1e-5, 1e-5), time_unit='ms',
                                              scalings=dict(eeg=1)), title="Diff - Entropy Onset")

# %% TRF split by musician
formal_training_years = [4,10,1.5,5,1.5,0,3,1,2,10,5,7,7,0,0,1,11,5,10,0,2,10,5,2,6,2]
np.quantile(formal_training_years,0.25)
np.quantile(formal_training_years,0.75)

nonmusician_ind = [i for i in (np.where(np.array(formal_training_years)<1.5)[0])]
musician_ind = [i for i in (np.where(np.array(formal_training_years)>=6.75)[0])]

subject_list_nonmusician = [subject_list[i] for i in (np.where(np.array(formal_training_years)<3.5)[0])]
subject_list_musician = [subject_list[i] for i in (np.where(np.array(formal_training_years)>=3.5)[0])]

trf_ori_AM_non_ave = np.mean(trf_ori_both[nonmusician_ind], axis=0)
trf_ori_AM_non_se = np.std(trf_ori_both[nonmusician_ind], axis=0)/np.sqrt(len(subject_list_nonmusician))
trf_ori_AM_mus_ave = np.mean(trf_ori_both[musician_ind], axis=0)
trf_ori_AM_mus_se = np.std(trf_ori_both[musician_ind], axis=0)/np.sqrt(len(subject_list_musician))

trf_chimera_AM_non_ave = np.mean(trf_chimera_both[nonmusician_ind], axis=0)
trf_chimera_AM_non_se = np.std(trf_chimera_both[nonmusician_ind], axis=0)/np.sqrt(len(subject_list_nonmusician))
trf_chimera_AM_mus_ave = np.mean(trf_chimera_both[musician_ind], axis=0)
trf_chimera_AM_mus_se = np.std(trf_chimera_both[musician_ind], axis=0)/np.sqrt(len(subject_list_musician))

regressors = ['spectral flux','note onset','Sp','Ep','St','Et']
colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']

reg=0
chi_n=1

for reg in range(len(regressors)):
#    for chi_n in range(32):
    for chi_n in [1]:
        fig = plt.figure()
        fig.set_size_inches(3,3)
        plt.plot(time*1e3, trf_ori_AM_mus_ave[reg,:,chi_n],c=colors[reg], linewidth=2, linestyle='solid', label="Original-Musician")
        plt.fill_between(time*1e3, trf_ori_AM_mus_ave[reg,:,chi_n]-trf_ori_AM_mus_se[reg,:,chi_n], trf_ori_AM_mus_ave[reg,:,chi_n]+trf_ori_AM_mus_se[reg,:,chi_n], color=colors[reg], alpha=0.8)
        plt.plot(time*1e3, trf_ori_AM_non_ave[reg,:,chi_n],c='k', linewidth=2, linestyle='solid', label="Original-Nonmusician")
        plt.fill_between(time*1e3, trf_ori_AM_non_ave[reg,:,chi_n]-trf_ori_AM_non_se[reg,:,chi_n], trf_ori_AM_non_ave[reg,:,chi_n]+trf_ori_AM_non_se[reg,:,chi_n], color='k', alpha=0.5)
        
        plt.plot(time*1e3, trf_chimera_AM_mus_ave[reg,:,chi_n], c=colors[reg], linewidth=1, linestyle='dashed', label="Chimeric-Musician")
        plt.fill_between(time*1e3, trf_chimera_AM_mus_ave[reg,:,chi_n]-trf_chimera_AM_mus_se[reg,:,chi_n], trf_chimera_AM_mus_ave[reg,:,chi_n]+trf_chimera_AM_mus_se[reg,:,chi_n], color=colors[reg],alpha=0.8)
        plt.plot(time*1e3, trf_chimera_AM_non_ave[reg,:,chi_n], c='k', linewidth=1, linestyle='dashed', label="Chimeric-Nonmusician")
        plt.fill_between(time*1e3, trf_chimera_AM_non_ave[reg,:,chi_n]-trf_chimera_AM_non_se[reg,:,chi_n], trf_chimera_AM_non_se[reg,:,chi_n]+trf_chimera_AM_non_ave[reg,:,chi_n], color='k',alpha=0.5)
        
        #plt.fill_between(time, -5e-5, 9e-5, where=(sig_time[chi_n,reg,:]==True), color='grey', ec=None, linewidth=0, alpha=0.3, zorder=2)
        plt.xlim(-100,700)
        plt.ylim(-2e-5, 4e-5)
        plt.grid()
        plt.title(regressors[reg])
        plt.legend(fontsize=10)
        plt.ylabel("Magnitude (AU)")
        plt.xlabel("Time (ms)")
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tight_layout()
        plt.show()
        plt.savefig(figure_path+'zscore_reg_individual_musicianship_quantile'+regressors[reg]+'_'+channel_names[chi_n]+'.svg', dpi=dpi, format='svg')


# %% The big Linear model
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pingouin
from scipy.stats import pearsonr

df = pd.DataFrame(columns=['subject','musician','category','group','model','preference','r'])

subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

formal_training_years = [4,10,1.5,5,1.5,0,3,1,2,10,5,7,7,0,0,1,11,5,10,0,2,10,5,2,6,2]
prefer_df = pd.read_pickle('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/preference.pkl')


for si in range(len(subject_list)):
    ms = float(formal_training_years[si])
    
    data_root = exp_path+'subject'+subject_list[si][7:] # EEG-BIDS root path
    eeg_data = read_hdf5(data_root+'/'+subject_list[si]+'_32chn_eeg_125hz_hp1_lp8_ICA_eye.hdf5')
    ori_epochs_df = eeg_data['ori_epochs_df']
    chimera_epochs_df = eeg_data['chimera_epochs_df']
    
    for ci in ['original','chimera']:
        for ei in range(66):
            if ci == 'original':
                f_name = ori_epochs_df.loc[ei, 'name'][:-4]
                pr = prefer_df[(prefer_df['fname']==f_name)&(prefer_df['subject']==subject_list[si])]['preference'].values[0]
                for mi in ['A','AM','LTM','STM','both']:
                    if mi == 'A':
                        r = ori_A_correlation[si,ei]
                        g = set_num_ori[si,ei]
                    elif mi == 'AM':
                        r = ori_AM_correlation[si,ei]
                        g = set_num_ori[si,ei]
                    elif mi == 'LTM':
                        r = ori_LTM_correlation[si,ei]
                        g = set_num_ori[si,ei]
                    elif mi == 'STM':
                        r = ori_STM_correlation[si,ei]
                        g = set_num_ori[si,ei]
                    elif mi == 'both':
                        r = ori_both_correlation[si,ei]
                        g = set_num_ori[si,ei]
                    
                    df = df._append({'subject':si, 'musician':ms, 'category':ci, 'group':g, 'model':mi, 'prefer':pr, 'r':r},ignore_index=True)

            elif ci == 'chimera':
                f_name = chimera_epochs_df.loc[ei, 'name'][:-4]
                pr = prefer_df[(prefer_df['fname']==f_name)&(prefer_df['subject']==subject_list[si])]['preference'].values[0]
                for mi in ['A','AM','LTM','STM','both']:
                    if mi == 'A':
                        r = chimera_A_correlation[si,ei]
                        g = set_num_chimera[si,ei]
                    elif mi == 'AM':
                        r = chimera_AM_correlation[si,ei]
                        g = set_num_chimera[si,ei]
                    elif mi == 'LTM':
                        r = chimera_LTM_correlation[si,ei]
                        g = set_num_chimera[si,ei]
                    elif mi == 'STM':
                        r = chimera_STM_correlation[si,ei]
                        g = set_num_chimera[si,ei]
                    elif mi == 'both':
                        r = chimera_both_correlation[si,ei]
                        g = set_num_chimera[si,ei]
                    
                    df = df._append({'subject':si, 'musician':ms, 'category':ci, 'group':g, 'model':mi, 'prefer':pr, 'r':r},ignore_index=True)

df.to_csv(exp_path+'expectation_effect_ltm_stm_both_nb_pitch_ioi_linear_model_zscore_reg_individual.csv')

# %%
from scipy.stats import pearsonr
df = pd.read_csv(exp_path+'expectation_effect_ltm_stm_both_nb_pitch_ioi_linear_model_zscore_reg_individual.csv')

df["random_eff"] = df["subject"].astype(str) + "+" + df["group"].astype(int).astype(str)
# df = pd.get_dummies(df, columns=['category'], drop_first=True)
# df['category_chimera'] = 1-df['category_original']


formula = "r ~ C(category, Treatment(reference='original')) + C(model) + C(category, Treatment(reference='original'))*C(model) + musician + prefer + C(category, Treatment(reference='original'))*musician + C(category, Treatment(reference='original'))*prefer"
r_lm = smf.mixedlm(formula, df, groups="random_eff").fit()
print(r_lm.summary())

pairwise = pingouin.pairwise_ttests(data=df, dv="r", subject="random_eff", within=["category","model"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
print(pairwise.round(3))
pairwise.columns
pairwise_arr = np.array(pairwise)

subject_group = df.groupby(['subject','category','model','musician'])
mean_subject_group = subject_group.mean(numeric_only=True).reset_index()
mean_ori_AM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'original') & (mean_subject_group['model'] == 'AM')]['r'])
mean_ori_A_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'original') & (mean_subject_group['model'] == 'A')]['r'])
mean_ori_LTM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'original') & (mean_subject_group['model'] == 'LTM')]['r'])
mean_ori_STM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'original') & (mean_subject_group['model'] == 'STM')]['r'])
mean_ori_both_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'original') & (mean_subject_group['model'] == 'both')]['r'])

mean_chimera_AM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'chimera') & (mean_subject_group['model'] == 'AM')]['r'])
mean_chimera_A_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'chimera') & (mean_subject_group['model'] == 'A')]['r'])
mean_chimera_STM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'chimera') & (mean_subject_group['model'] == 'STM')]['r'])
mean_chimera_LTM_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'chimera') & (mean_subject_group['model'] == 'LTM')]['r'])
mean_chimera_both_sub = np.array(mean_subject_group[(mean_subject_group['category'] == 'chimera') & (mean_subject_group['model'] == 'both')]['r'])

pearsonr(mean_subject_group[(mean_subject_group['model'] == 'A')]['r'],mean_subject_group[(mean_subject_group['model'] == 'A')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'STM')]['r'],mean_subject_group[(mean_subject_group['model'] == 'AM')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'LTM')]['r'],mean_subject_group[(mean_subject_group['model'] == 'LTM')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'both')]['r'],mean_subject_group[(mean_subject_group['model'] == 'both')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'AM')]['r'],mean_subject_group[(mean_subject_group['model'] == 'STM')]['musician'])

# Plot delta_r correlation musicianship
plt.scatter(mean_subject_group[mean_subject_group['model']=="both"]['musician'],mean_subject_group[mean_subject_group['model']=="both"]['r'])
# Correlation with fitted line
m, b = np.polyfit(mean_subject_group[mean_subject_group['model']=="both"]['musician'], mean_subject_group[mean_subject_group['model']=="both"]['r'], 1)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5,3)
plt.scatter(mean_subject_group[mean_subject_group['model']=="both"]['musician'],mean_subject_group[mean_subject_group['model']=="both"]['r'])
plt.plot(mean_subject_group[mean_subject_group['model']=="both"]['musician'],m*np.array(mean_subject_group[mean_subject_group['model']=="both"]['musician'])+b,c='k')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel("Prediction Correlation \n $AM_{both}$ (r)", fontsize=10)
plt.xlabel("Formal music training years")
#plt.title("Original Note onset P2 (Fz)")
plt.grid(alpha=0.5)
plt.show()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+"r_both_musician_corr.png",dpi=dpi,format='png', bbox_inches='tight')

# Correlation with fitted line
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'both') & (mean_subject_group['category'] == 'original')]['r'],mean_subject_group[(mean_subject_group['model'] == 'both')& (mean_subject_group['category'] == 'original')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'both') & (mean_subject_group['category'] == 'chimera')]['r'],mean_subject_group[(mean_subject_group['model'] == 'both')& (mean_subject_group['category'] == 'chimera')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'A') & (mean_subject_group['category'] == 'original')]['r'],mean_subject_group[(mean_subject_group['model'] == 'A')& (mean_subject_group['category'] == 'original')]['musician'])
pearsonr(mean_subject_group[(mean_subject_group['model'] == 'A') & (mean_subject_group['category'] == 'chimera')]['r'],mean_subject_group[(mean_subject_group['model'] == 'A')& (mean_subject_group['category'] == 'chimera')]['musician'])



m, b = np.polyfit(mean_subject_group[(mean_subject_group['model']=="both") & (mean_subject_group['category'] == 'original')]['musician'], mean_subject_group[(mean_subject_group['model']=="both") & (mean_subject_group['category'] == 'original')]['r'], 1)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5,3)
plt.scatter(mean_subject_group[mean_subject_group['model']=="both"]['musician'],mean_subject_group[mean_subject_group['model']=="both"]['r'])
plt.plot(mean_subject_group[mean_subject_group['model']=="both"]['musician'],m*np.array(mean_subject_group[mean_subject_group['model']=="both"]['musician'])+b,c='k')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel("Prediction Correlation \n $AM_{both}$ (r)", fontsize=10)
plt.xlabel("Formal music training years")
#plt.title("Original Note onset P2 (Fz)")
plt.grid(alpha=0.5)
plt.show()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+"r_both_musician_corr.png",dpi=dpi,format='png', bbox_inches='tight')


_, p_corr = multitest.fdrcorrection([0.0002611494554925402,0.0002647005834143278,0.0003639720243759706,0.0003374281550332811], method='i')

set_group = df.groupby(['random_eff','category','model','musician'])
mean_set_group = set_group.mean().reset_index()
mean_ori_AM = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'AM')]['r'])
mean_ori_AM_group = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'AM')]['random_eff'])
mean_ori_AM_musician  = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'AM')]['musician'])
mean_ori_A = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'A')]['r'])
mean_ori_LTM = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'LTM')]['r'])
mean_ori_STM = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'STM')]['r'])
mean_ori_both = np.array(mean_set_group[(mean_set_group['category'] == 'original') & (mean_set_group['model'] == 'both')]['r'])

mean_chimera_AM = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'AM')]['r'])
mean_chimera_AM_group = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'AM')]['random_eff'])
mean_chimera_A = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'A')]['r'])
mean_chimera_STM = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'STM')]['r'])
mean_chimera_LTM = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'LTM')]['r'])
mean_chimera_both = np.array(mean_set_group[(mean_set_group['category'] == 'chimera') & (mean_set_group['model'] == 'both')]['r'])


stats.ttest_rel(mean_ori_AM, mean_ori_A)
stats.ttest_rel(mean_chimera_AM, mean_chimera_A)
stats.ttest_rel(mean_ori_AM, mean_chimera_AM)
stats.ttest_rel(mean_ori_A, mean_chimera_A)
stats.ttest_rel(mean_ori_AM-mean_ori_A, mean_chimera_AM-mean_chimera_A)
#Plot
y=[mean_ori_A.mean(),mean_chimera_A.mean(),
   mean_ori_STM.mean(),mean_chimera_STM.mean(),
   mean_ori_LTM.mean(),mean_chimera_LTM.mean(),
   mean_ori_AM.mean(),mean_chimera_AM.mean(),
   mean_ori_both.mean(),mean_chimera_both.mean(),]
yerr = [mean_ori_A.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_A.std()/np.sqrt(len(mean_ori_AM_group)),
        mean_ori_STM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_STM.std()/np.sqrt(len(mean_ori_AM_group)),
        mean_ori_LTM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_LTM.std()/np.sqrt(len(mean_ori_AM_group)),
        mean_ori_AM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_AM.std()/np.sqrt(len(mean_ori_AM_group)),
        mean_ori_both.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_both.std()/np.sqrt(len(mean_ori_AM_group))]

dpi = 300
plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams["font.family"] = "Arial"

figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/paper_figs/'
barWidth = 0.15
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1[0], mean_ori_A.mean(), color='grey', width=barWidth, label='A',zorder=3)
plt.bar(br1[1], mean_chimera_A.mean(), edgecolor='grey', fill=False, linewidth=2, width=barWidth,zorder=3)
plt.bar(br2[0], mean_ori_STM.mean(), color='C0', width=barWidth, label='$AM_{STM}$',zorder=3)
plt.bar(br2[1], mean_chimera_STM.mean(),edgecolor='C0', fill=False, linewidth=2, width=barWidth,zorder=3)
plt.bar(br3[0], mean_ori_LTM.mean(), color='C1', width=barWidth, label='$AM_{LTM}$',zorder=3)
plt.bar(br3[1], mean_chimera_LTM.mean(), edgecolor='C1', fill=False, linewidth=2, width=barWidth,zorder=3)
plt.bar(br4[0], mean_ori_AM.mean(), color='C2', width=barWidth, label='$AM_{STM+LTM}$',zorder=3)
plt.bar(br4[1], mean_chimera_AM.mean(), edgecolor='C2', fill=False, linewidth=2, width=barWidth,zorder=3)
plt.bar(br5[0], mean_ori_both.mean(), color='C3', width=barWidth, label='$AM_{both}$',zorder=3)
plt.bar(br5[1], mean_chimera_both.mean(), edgecolor='C3', fill=False, linewidth=2, width=barWidth,zorder=3)
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1],br4[0],br4[1],br5[0],br5[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus Category')
plt.ylabel("Prediction Accuracy \n(r)")
#plt.ylim(0.05, 0.225)
plt.ylim(0.1, 0.125)
plt.xlim(-0.25, 1.8)
plt.xticks([r + barWidth*1.5 for r in range(2)], ['Original', 'Chimeric'])
# for si in range(len(subject_list)):
#     plt.plot([br1[0],br2[0], br3[0],br4[0]],[mean_ori_A_sub[si],mean_ori_STM_sub[si],mean_ori_LTM_sub[si],mean_ori_AM_sub[si]], ".-", markersize=2, linewidth=0.5, c='grey',zorder=5,alpha=0.5)
#     plt.plot([br1[1],br2[1],br3[1],br4[1]],[mean_chimera_A_sub[si],mean_chimera_STM_sub[si],mean_chimera_LTM_sub[si],mean_chimera_AM[si]], ".-", markersize=2,linewidth=0.5, c='grey',zorder=5,alpha=0.5)

lg = plt.legend(fontsize=10, loc=(1.1,0.7))
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'model_prediction_accuracy_r_reg1_zscore_reg_individual.png', dpi=dpi, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% Delta r model

delta_r_df = pd.DataFrame(columns=["model","category","delta_r","random_eff","musician"])

delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["STM"]*len(mean_ori_AM_group)),"category":["original"]*len(mean_ori_AM_group), "random_eff":list(mean_ori_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_ori_STM-mean_ori_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["LTM"]*len(mean_ori_AM_group)),"category":["original"]*len(mean_ori_AM_group), "random_eff":list(mean_ori_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_ori_LTM-mean_ori_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["both"]*len(mean_ori_AM_group)),"category":["original"]*len(mean_ori_AM_group), "random_eff":list(mean_ori_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_ori_both-mean_ori_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["AM"]*len(mean_ori_AM_group)),"category":["original"]*len(mean_ori_AM_group), "random_eff":list(mean_ori_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_ori_AM-mean_ori_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["STM"]*len(mean_ori_AM_group)),"category":["chimera"]*len(mean_ori_AM_group), "random_eff":list(mean_chimera_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_chimera_STM-mean_chimera_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["LTM"]*len(mean_ori_AM_group)),"category":["chimera"]*len(mean_ori_AM_group), "random_eff":list(mean_chimera_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_chimera_LTM-mean_chimera_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["both"]*len(mean_ori_AM_group)),"category":["chimera"]*len(mean_ori_AM_group), "random_eff":list(mean_ori_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_chimera_both-mean_chimera_A)})])
delta_r_df = pd.concat([delta_r_df, pd.DataFrame({"model":np.array(["AM"]*len(mean_ori_AM_group)),"category":["chimera"]*len(mean_ori_AM_group), "random_eff":list(mean_chimera_AM_group),"musician":list(mean_ori_AM_musician),
                                "delta_r":list(mean_chimera_AM-mean_chimera_A)})])

delta_r_df['subject'] = delta_r_df['random_eff'].str.split("+", expand=True)[0].astype(str)
delta_r_df.to_csv(exp_path+'expectation_effect_ltm_stm_nb_pitch_ioi_linear_model_zscore_reg_individual_delta_r.csv')

delta_r_df = pd.read_csv(exp_path+'expectation_effect_ltm_stm_nb_pitch_ioi_linear_model_zscore_reg_individual_delta_r.csv')

formula = "delta_r ~ C(category,Treatment(reference='original')) + C(model,Treatment(reference='STM')) + C(category, Treatment(reference='original'))*C(model,Treatment(reference='STM')) + musician + musician*C(category, Treatment(reference='original'))"
delta_r_lm = smf.mixedlm(formula, delta_r_df, groups="random_eff").fit()
print(delta_r_lm.summary())
# post-hoc
pairwise_delta_r = pingouin.pairwise_ttests(data=delta_r_df, dv="delta_r", subject="random_eff", within=["model","category"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
print(pairwise.round(3))
pairwise.columns
pairwise_arr_delta_r = np.array(pairwise_delta_r)

delta_r_subject_group = delta_r_df.groupby(['subject','category','model'])
delta_r_mean_subject_group = delta_r_subject_group.mean(numeric_only=True).reset_index()
mean_ori_AM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'original') & (delta_r_mean_subject_group['model'] == 'AM')]['delta_r'])
mean_ori_LTM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'original') & (delta_r_mean_subject_group['model'] == 'LTM')]['delta_r'])
mean_ori_STM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'original') & (delta_r_mean_subject_group['model'] == 'STM')]['delta_r'])
mean_ori_both_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'original') & (delta_r_mean_subject_group['model'] == 'both')]['delta_r'])

mean_chimera_AM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'chimera') & (delta_r_mean_subject_group['model'] == 'AM')]['delta_r'])
mean_chimera_STM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'chimera') & (delta_r_mean_subject_group['model'] == 'STM')]['delta_r'])
mean_chimera_LTM_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'chimera') & (delta_r_mean_subject_group['model'] == 'LTM')]['delta_r'])
mean_chimera_both_subject = np.array(delta_r_mean_subject_group[(delta_r_mean_subject_group['category'] == 'chimera') & (delta_r_mean_subject_group['model'] == 'both')]['delta_r'])


delta_r_set_group = delta_r_df.groupby(['random_eff','category','model','musician'])
delta_r_mean_set_group = delta_r_set_group.mean(numeric_only=True).reset_index()
mean_ori_AM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'AM')]['delta_r'])
mean_ori_AM_group = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'AM')]['random_eff'])
mean_ori_AM_musician  = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'AM')]['musician'])
mean_ori_LTM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'LTM')]['delta_r'])
mean_ori_STM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'STM')]['delta_r'])
mean_ori_both = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'original') & (delta_r_mean_set_group['model'] == 'both')]['delta_r'])

mean_chimera_AM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'chimera') & (delta_r_mean_set_group['model'] == 'AM')]['delta_r'])
mean_chimera_AM_group = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'chimera') & (delta_r_mean_set_group['model'] == 'AM')]['random_eff'])
mean_chimera_STM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'chimera') & (delta_r_mean_set_group['model'] == 'STM')]['delta_r'])
mean_chimera_LTM = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'chimera') & (delta_r_mean_set_group['model'] == 'LTM')]['delta_r'])
mean_chimera_both = np.array(delta_r_mean_set_group[(delta_r_mean_set_group['category'] == 'chimera') & (delta_r_mean_set_group['model'] == 'both')]['delta_r'])



# Plot delta_r correlation musicianship
pearsonr(delta_r_df['musician'],delta_r_df['delta_r'])
pearsonr(delta_r_mean_subject_group['musician'], delta_r_mean_subject_group['delta_r'])
pearsonr(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'], delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['delta_r'])

plt.scatter(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'],delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['delta_r'])
# Correlation with fitted line
m, b = np.polyfit(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'], delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['delta_r'], 1)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5,3)
plt.scatter(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'], delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['delta_r'])
plt.plot(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'],m*np.array(delta_r_mean_subject_group[delta_r_mean_subject_group['model']=="STM"]['musician'])+b,c='k')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel("Melodic Expectation Effect \n $AM_{STM}$-A (Δr)", fontsize=10)
plt.xlabel("Formal music training years")
#plt.title("Original Note onset P2 (Fz)")
plt.grid(alpha=0.5)
plt.show()
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+"delta_r_STM_musician_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

# #Plot
# y=[mean_ori_STM.mean(),mean_chimera_STM.mean(),
#    mean_ori_LTM.mean(),mean_chimera_LTM.mean(),
#    mean_ori_AM.mean(),mean_chimera_AM.mean(),]
# yerr = [mean_ori_STM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_STM.std()/np.sqrt(len(mean_ori_AM_group)),
#         mean_ori_LTM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_LTM.std()/np.sqrt(len(mean_ori_AM_group)),
#         mean_ori_AM.std()/np.sqrt(len(mean_ori_AM_group)), mean_chimera_AM.std()/np.sqrt(len(mean_ori_AM_group))]

# dpi = 300
# plt.rc('axes', titlesize=10, labelsize=10)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.rcParams["font.family"] = "Arial"

# figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/paper_figs/'
# barWidth = 0.2
# br1 = np.arange(2)
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]

# fig = plt.figure(dpi=dpi)
# fig.set_size_inches(6, 3)

# plt.bar(br1[0], mean_ori_STM.mean(), color='C1', width=barWidth, label='$AM_{STM}$-A',zorder=3)
# plt.bar(br1[1], mean_chimera_STM.mean(), color='C1', width=barWidth,zorder=3)
# plt.bar(br2[0], mean_ori_LTM.mean(), color='C2', width=barWidth, label='$AM_{LTM}$-A',zorder=3)
# plt.bar(br2[1], mean_chimera_LTM.mean(), color='C2', width=barWidth,zorder=3)
# plt.bar(br3[0], mean_ori_AM.mean(),  color='C3', width=barWidth, label='$AM_{STM+LTM}$-A',zorder=3)
# plt.bar(br3[1], mean_chimera_AM.mean(), color='C3', width=barWidth,zorder=3)
# plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
# plt.xlabel('Stimulus')
# plt.ylabel("Melodic Expectation Effect \n (Δr)")
# #plt.ylim(-0, 0.045)
# #plt.ylim(0.1, 0.125)
# plt.xlim(-0.25, 1.8)
# plt.xticks([r + barWidth*1.2 for r in range(2)], ['Original', 'Chimera'])
# for si in range(len(subject_list)):
#     #plt.plot([br1[0],br2[0], br3[0]],[cc_music[0][si],cc_music[1][si], cc_music[2][si]], ".-", markersize=2, linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
#     plt.plot([br1[0],br2[0], br3[0]],[mean_ori_STM_subject[si],mean_ori_LTM_subject[si], mean_ori_AM_subject[si]], ".-", markersize=2, linewidth=0.5, c='grey',zorder=5,alpha=0.5)
#     #plt.plot([br1[1],br2[1], br3[1]],[cc_speech[0][si],cc_speech[1][si], cc_speech[2][si]], ".-", markersize=2,linewidth=0.5, c='k',zorder=4,alpha=0.3+0.03*si)
#     plt.plot([br1[1],br2[1],br3[1]],[mean_chimera_STM_subject[si],mean_chimera_LTM_subject[si],mean_chimera_AM_subject[si]], ".-", markersize=2,linewidth=0.5, c='grey',zorder=5,alpha=0.5)

# lg = plt.legend(fontsize=10, loc=(1.1,0.7))
# plt.grid(alpha=0.5,zorder=0)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# #plt.savefig(figure_path+'predict_real_corr_class_200ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
# #plt.savefig(figure_path+'predict_real_corr_class_15ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
# plt.savefig(figure_path+'model_prediction_accuracy_delta_r.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')


# Plot within model
y=[mean_ori_STM.mean(),mean_ori_LTM.mean(),mean_ori_AM.mean(),mean_ori_both.mean(),
   mean_chimera_STM.mean(),mean_chimera_LTM.mean(), mean_chimera_AM.mean(),mean_chimera_both.mean()]
yerr = [mean_ori_STM.std()/np.sqrt(len(mean_ori_AM_group)), mean_ori_LTM.std()/np.sqrt(len(mean_ori_AM_group)),mean_ori_AM.std()/np.sqrt(len(mean_ori_AM_group)),mean_ori_both.std()/np.sqrt(len(mean_ori_AM_group)),
        mean_chimera_STM.std()/np.sqrt(len(mean_ori_AM_group)),mean_chimera_LTM.std()/np.sqrt(len(mean_ori_AM_group)),mean_chimera_AM.std()/np.sqrt(len(mean_ori_AM_group)),mean_chimera_both.std()/np.sqrt(len(mean_ori_AM_group))]

dpi = 300
plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams["font.family"] = "Arial"

figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/paper_figs/'
barWidth = 0.15
br1 = np.arange(4)
br2 = [x + barWidth for x in br1]

fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)

plt.bar(br1[0], mean_ori_STM.mean(), color='C0', width=barWidth, label='Original',zorder=3)
plt.bar(br1[1], mean_ori_LTM.mean(), color='C1', width=barWidth,zorder=3)
plt.bar(br1[2], mean_ori_AM.mean(), color='C2', width=barWidth,zorder=3)
plt.bar(br1[3], mean_ori_both.mean(), color='C3', width=barWidth,zorder=3)

plt.bar(br2[0], mean_chimera_STM.mean(), edgecolor='C0', linewidth=2, fill=False, width=barWidth, label='Chimeric',zorder=3)
plt.bar(br2[1], mean_chimera_LTM.mean(), edgecolor='C1', linewidth=2, fill=False, width=barWidth, zorder=3)
plt.bar(br2[2], mean_chimera_AM.mean(), edgecolor='C2', linewidth=2, fill=False, width=barWidth, zorder=3)
plt.bar(br2[3], mean_chimera_both.mean(), edgecolor='C3', linewidth=2, fill=False, width=barWidth, zorder=3)

plt.errorbar([br1[0],br1[1],br1[2],br1[3], br2[0],br2[1],br2[2],br2[3]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
#plt.xlabel('Model')
plt.ylabel("Melodic Expectation Effect \n (Δr)")
#plt.ylim(-0, 0.045)
plt.ylim(-0.0055, 0.015)
plt.xlim(-0.25, 3.5)
plt.xticks([r + barWidth/2 for r in range(4)], ['$AM_{STM}$-A','$AM_{LTM}$-A','$AM_{STM+LTM}$-A', '$AM_{both}$-A'])
for si in range(len(subject_list)):
    plt.plot([br1[0],br2[0]],[mean_ori_STM_subject[si],mean_chimera_STM_subject[si]], ".-", markersize=2, linewidth=0.5, c='grey',zorder=5,alpha=0.5)
    plt.plot([br1[1],br2[1]],[mean_ori_LTM_subject[si],mean_chimera_LTM_subject[si]], ".-", markersize=2,linewidth=0.5, c='grey',zorder=5,alpha=0.5)
    plt.plot([br1[2],br2[2]],[mean_ori_AM_subject[si],mean_chimera_AM_subject[si]], ".-", markersize=2,linewidth=0.5, c='grey',zorder=5,alpha=0.5)
    plt.plot([br1[3],br2[3]],[mean_ori_both_subject[si],mean_chimera_both_subject[si]], ".-", markersize=2,linewidth=0.5, c='grey',zorder=5,alpha=0.5)


lg = plt.legend(fontsize=10, loc=(1.1,0.7))
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.savefig(figure_path+'predict_real_corr_class_200ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
#plt.savefig(figure_path+'predict_real_corr_class_15ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
plt.savefig(figure_path+'model_prediction_accuracy_delta_r_subject.png', dpi=dpi, format='png', bbox_extra_artists=(lg,), bbox_inches='tight')