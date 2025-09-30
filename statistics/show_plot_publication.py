#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 04:32:52 2024

@author: tshan
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
from cycler import cycler
import os
import json
import pingouin
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
figure_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/paper_figs/'
corpus_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/'
#%% FIG SETTING
dpi = 300
plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rcParams["font.family"] = "Arial"
# %% Stim stats
reg_df = pd.read_pickle(corpus_path+"regressor/125hz/downbeat_trf_reg.pkl")
reg_df['duration'].mean()
reg_df['duration'].min()
reg_df['duration'].max()
reg_df['duration'].sum()/60
# %% Preference plot
df = pd.read_pickle('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/preference.pkl')
orig_mean = df[df['category']=='original']['preference'].mean()
orig_sem = df[df['category']=='original']['preference'].std()/np.sqrt(len(df[df['category']=='original']))
chimera_mean = df[df['category']=='chimeric']['preference'].mean()
chimera_sem = df[df['category']=='chimeric']['preference'].std()/np.sqrt(len(df[df['category']=='chimeric']))


df["random_eff"] = df["subject"].astype(str) + "+" + df["group"].astype(int).astype(str)
df_mean = df.groupby(["category","random_eff"])['preference'].mean().reset_index()

df_ori = df_mean[df_mean["category"]=="original"].set_index('random_eff')['preference']
df_chimera = df_mean[df_mean["category"]=="chimeric"].set_index('random_eff')['preference']

stats.ttest_rel(df_ori, df_chimera)


df_mean = df.groupby(["category","subject"])['preference'].mean().reset_index()
df_ori = df_mean[df_mean["category"]=="original"].set_index('subject')['preference']
df_chimera = df_mean[df_mean["category"]=="chimeric"].set_index('subject')['preference']

df_ori_mean = np.mean(df_ori)
df_ori_sem = np.mean(df_ori)/np.sqrt(27)
df_chimera_mean = np.mean(df_chimera)
df_chimera_sem = np.mean(df_chimera)/np.sqrt(27)

stats.ttest_rel(df_ori, df_chimera)

dpi=300
plt.figure()
barWidth = 0.25
br1 = np.arange(0.5,1.5, 0.5)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(2.5, 3)
plt.bar(br1[0], df_ori_mean, color='C0', width=barWidth, label='Original',zorder=3)
plt.bar(br1[1], df_chimera_mean, color='C1',label='Chimeric', width=barWidth,zorder=3)
plt.errorbar([br1[0],br1[1]], y=[df_ori_mean,df_chimera_mean], yerr=[df_ori_sem,df_chimera_sem],linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus Category')
plt.ylabel("Preference \nRating")
plt.xticks([r for r in br1], ['Original', 'Chimeric'])
plt.ylim(0, 5.5)
# plt.xlim(-0.25, 1.5)
#lg = plt.legend(fontsize=10, loc='upper right')
plt.grid(alpha=0.5,zorder=0)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.savefig(figure_path+'predict_real_corr_class_200ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
#plt.savefig(figure_path+'predict_real_corr_class_15ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')
plt.tight_layout()
plt.savefig(figure_path+'preference_dist.svg', dpi=dpi, format='svg')

# %% Expectation model

# %% Downbeat model
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
# %% Clustering
from scipy.stats import ttest_ind
import mne
from mne.channels import find_ch_adjacency, make_1020_channel_selections
from mne.stats import spatio_temporal_cluster_test, spatio_temporal_cluster_1samp_test

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
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
axes['Left'].set_title("Left")
axes['Midline'].set_title("Midline")
axes['Right'].set_title("Right")
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"original_onset.svg",format='svg',dpi=dpi)

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
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
plt.savefig(figure_path+"original_downbeat.svg",format='svg',dpi=dpi)

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
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
axes['Left'].set_title("Left")
axes['Midline'].set_title("Midline")
axes['Right'].set_title("Right")
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"chimera_onset.svg",format='svg',dpi=dpi)

######## Originl vs Chimera onset
X_ori_vs_chimera_onset = [Epochs_ori_onset.get_data().transpose(0,2,1),
     Epochs_chimera_onset.get_data().transpose(0,2,1),]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X_ori_vs_chimera_onset, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_vs_chimera_onset_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_vs_chimera_onset_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_ori_vs_chimera_onset = mne.combine_evoked([Epochs_ori_onset.average(), Epochs_chimera_onset.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked_ori_vs_chimera_onset.plot_joint(
    title="Original vs Chimera onset", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_ori_vs_chimera_onset.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
plt.savefig(figure_path+"original_vs_chimera_onset.svg",format='svg',dpi=dpi)


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
plt.savefig(figure_path+"chimera_db_pitch.svg",format='svg',dpi=dpi)

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
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
plt.savefig(figure_path+"Chimera_db_timing.svg", dpi=dpi, format='svg')

######## Originl vs Chimera downbeat
X_ori_vs_chimera_downbeat = [Epochs_ori_downbeat.get_data().transpose(0,2,1),
     Epochs_chimera_timing.get_data().transpose(0,2,1),]

tfce = dict(start=0, step=0.7) 
# Calculate statistical thresholds
t_obs, clusters, cluster_pv, h0 = spatio_temporal_cluster_test(
    X_ori_vs_chimera_downbeat, tfce, adjacency=adjacency, n_permutations=1000)  # a more standard number would be 1000+
ori_vs_chimera_downbeat_significant_points = cluster_pv.reshape(t_obs.shape).T < 0.05
print(str(ori_vs_chimera_downbeat_significant_points.sum()) + " points selected by TFCE ...")

# We need an evoked object to plot the image to be masked
evoked_ori_vs_chimera_downbeat = mne.combine_evoked([Epochs_ori_downbeat.average(), Epochs_chimera_timing.average()], weights=[1, -1])
time_unit = dict(time_unit="ms")
evoked_ori_vs_chimera_downbeat.plot_joint(
    title="Original vs Chimera dowbbeat", ts_args=time_unit, topomap_args=time_unit
)  # show difference wave

# Create ROIs by checking channel labels
selections = make_1020_channel_selections(evoked_ori_vs_chimera_downbeat.info, midline=['z','1','2'])

# Visualize the results
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
axes = {sel: ax for sel, ax in zip(selections, axes.ravel())}
evoked_ori_vs_chimera_downbeat.plot_image(
    axes=axes,
    group_by=selections,
    colorbar=False,
    show=False,
    mask=ori_vs_chimera_downbeat_significant_points,
    show_names="all",
    titles=None,
    **time_unit,
)
plt.colorbar(axes["Left"].images[-1], ax=list(axes.values()), shrink=0.3, label="µV")
plt.show()
plt.savefig(figure_path+"original_vs_chimera_downbeat.svg",format='svg',dpi=dpi)

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
fig, axes = plt.subplots(nrows=3, figsize=(8,8))
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
plt.savefig(figure_path+"Chimera_db_inter.svg", dpi=dpi, format='svg')

# %% Stacked TRF
from matplotlib.ticker import FixedLocator, FixedFormatter
chn=1
fig, axes = plt.subplots(6, sharex=True)
fig.set_size_inches(3.5,8)
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


axes[3].plot(lag, trf_chimera_all_ave[2,:,chn], c='C2', linewidth=1, linestyle='dashed', label="Chimera-Timing")
axes[3].fill_between(lag, trf_chimera_all_ave[2,:,chn]-trf_chimera_all_err[2,:,chn], trf_chimera_all_ave[2,:,chn]+trf_chimera_all_err[2,:,chn], color='C2',alpha=0.5)
axes[3].set_ylim(-1e-4,1e-4)
axes[3].fill_between(lag,-1e-4,1e-4, where=(chimera_timing_significant_points[chn]==True), color='gray', alpha=0.3)
axes[3].vlines(0,-1e-4,1e-4,color='k',linestyle="solid")
axes[3].vlines(185,-1e-4,1e-4,color='gray',linestyle="dashed")
axes[3].yaxis.set_major_locator(FixedLocator([-1e-4,1e-4]))


axes[4].plot(lag, trf_chimera_all_ave[1,:,chn], c='C1', linewidth=1, linestyle='dashed', label="Chimera-Pitch")
axes[4].fill_between(lag, trf_chimera_all_ave[1,:,chn]-trf_chimera_all_err[1,:,chn], trf_chimera_all_ave[1,:,chn]+trf_chimera_all_err[1,:,chn], color='C1',alpha=0.5)
axes[4].set_ylim(-1e-4,1e-4)
axes[4].fill_between(lag,-1e-4,1e-4, where=(chimera_pitch_significant_points[chn]==True), color='gray', alpha=0.3)
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
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_ori_onset.plot_topomap(times=[0.185], vlim=(-4.5, 4.5), scalings=1e4, units='AU', time_unit='ms',
                                    mask=ori_onset_significant_points, contours=0,
                                    mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"Original_onset_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')

# Original downbeat
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_ori_downbeat.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=ori_downbeat_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"Original_downbeat_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')

# Chimera Onset
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_onset.plot_topomap(times=[0.185], vlim=(-4.5, 4.5), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_onset_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_onset_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')

# Chimera pitch
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_pitch.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_pitch_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_pitch_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')

# Chimera timing
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_timing.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_timing_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_timing_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')


# Chimera interaction
fig.set_size_inches(3,3)
mask_params = dict(marker='o', markerfacecolor='w', markeredgecolor='k', linewidth=0, markersize=6)
fig = evoked_chimera_inter.plot_topomap(times=[0.185], vlim=(-1, 1), scalings=1e4, units='AU', time_unit='ms',
                              mask=chimera_inter_significant_points, contours=0,
                              mask_params=mask_params,colorbar=True)
fig.axes[0].set_title("")
fig.axes[1].set_title("")
plt.savefig(figure_path+"chimera_inter_topo.svg",dpi=dpi,format='svg', bbox_inches='tight')

# %% All channel correlation musicianship
from statsmodels.stats import multitest

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
fig.suptitle("Original Note onset P1", fontsize=10)
plt.savefig(figure_path+"Original_onset_P1amp_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

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
fig.suptitle("Original Note onset P2", fontsize=10)
plt.savefig(figure_path+"Original_onset_P2amp_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

# Correlation with fitted line
m, b = np.polyfit(formal_training_years, ori_onset_amplitude_p2_all[1], 1)
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5,3)
plt.scatter(formal_training_years, ori_onset_amplitude_p2_all[1])
plt.plot(formal_training_years,m*np.array(formal_training_years)+b,c='k')
plt.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
plt.ylabel("Amplitude (AU)", fontsize=10)
plt.xlabel("Formal music training years")
plt.title("Original Note onset P2 (Fz)")
plt.grid()
plt.show()
plt.savefig(figure_path+"Original_onset_P2amp_musicianship_corr_scatter.svg",dpi=dpi,format='svg', bbox_inches='tight')



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
fig.suptitle("Original Downbeat peak", fontsize=10)
plt.savefig(figure_path+"Original_downbeat_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

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
fig.suptitle("Chimera Note onset P1", fontsize=10)
plt.savefig(figure_path+"chimera_onset_P1amp_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

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
fig.suptitle("Chimera Note onset P2", fontsize=10)
plt.savefig(figure_path+"chimera_onset_P2amp_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')

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
fig.suptitle("Chimera Timing Peak", fontsize=10)
plt.savefig(figure_path+"chimera_timing_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')


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
fig.suptitle("Chimera Interaction Peak", fontsize=10)
plt.savefig(figure_path+"chimera_inter_musicianship_corr.svg",dpi=dpi,format='svg', bbox_inches='tight')
    
# %% ABR
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


data_path = '/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicABR/MusicExp/'
ABR_ori = np.zeros((len(subject_list),8000))
ABR_chimera = np.zeros((len(subject_list),8000))
for subject, subject_id in zip(subject_list, subject_ids):
    data_root = exp_path+"subject_{0:03d}".format(subject_id)
    ABR_data = read_hdf5(data_root+'/'+subject+'_ABR_data.hdf5')
    ABR_ori[subject_id-2] = ABR_data['abr_bp']['ori']
    ABR_chimera[subject_id-2] = ABR_data['abr_bp']['chimera']
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

fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ABR_ori_ave, c="C0", linewidth=2, linestyle='solid', label="Original")
plt.fill_between(lags, ABR_ori_ave-ABR_ori_se, ABR_ori_ave+ABR_ori_se, color="C0", alpha=0.8)
plt.plot(lags, ABR_chimera_ave, c="C1", linewidth=1, linestyle='solid', label="Chimeric")
plt.fill_between(lags, ABR_chimera_ave-ABR_chimera_se, ABR_chimera_ave+ABR_chimera_se, color="C1", alpha=0.5)
plt.xlim(-10, 30)
plt.ylim(-40, 65)
plt.xlabel("Time (ms)")
plt.ylabel("Magnitude (AU)")
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(figure_path+'ABR_results.svg', dpi=dpi, format='svg')