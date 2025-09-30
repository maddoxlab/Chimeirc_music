#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:23:03 2023

@author: tshan@urmc-sh.rochester.edu
"""
"""
Created on Thu Apr 27 12:22:02 2023

@author: tshan@urmc-sh.rochester.edu
"""

import scipy.signal as signal
import mne
import os
from mne.preprocessing import ICA, create_eog_epochs

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

# %% Get EEG data, save into one file
subject_list = ['chimera_001','chimera_002', 'chimera_003','chimera_004',
                'chimera_005','chimera_006','chimera_007','chimera_008',
                'chimera_009','chimera_010','chimera_011','chimera_012',
                'chimera_013','chimera_014','chimera_015','chimera_016',
                'chimera_017','chimera_018','chimera_019','chimera_020',
                'chimera_021','chimera_022','chimera_023','chimera_024',
                'chimera_025','chimera_pilot_1_64chn','chimera_pilot_2_64chn']

subject = 'chimera_pilot_2_64chn'

# %% Load EEG data
data_root = exp_path+'subject'+subject[7:] # EEG-BIDS root path
if subject=='chimera_001' or subject=='chimera_003':
    eeg_vhdr = data_root+'/'+subject+'_2.vhdr'
elif subject=='chimera_010':
    eeg_vhdr = data_root+'/'+subject+'_3.vhdr'
else:
    eeg_vhdr = data_root+'/'+subject+'.vhdr'
for file in os.listdir(data_root):
    if file.endswith(".csv"):
        dataframe_path = data_root +"/"+ file
        
# read raw EEG
eeg_raw = mne.io.read_raw_brainvision(eeg_vhdr, preload=True)
eeg_raw = eeg_raw.pick(channel_names)
# Resample
eeg_raw.resample(eeg_fs_down)
# Re-reference
eeg_raw.set_eeg_reference(ref_channels=ref_channels)
# Filtering
eeg_raw.filter(1,8)
# Set montage
montage = mne.channels.make_standard_montage('standard_1020')
eeg_raw.set_montage(montage)
eeg_raw.plot()
# %% Show EOG artifact ERP
eeg_raw.add_reference_channels(['EOG 001'])
eeg_raw.set_channel_types({'EOG 001':'eog'})
eeg_raw._data[eeg_raw.ch_names.index('EOG 001')] = (eeg_raw._data[eeg_raw.ch_names.index('Fp1')]+eeg_raw._data[eeg_raw.ch_names.index('Fp2')])/2
eog_evoked = create_eog_epochs(eeg_raw).average()
#eog_evoked.apply_baseline(baseline=(None, -0.2))
eog_evoked.plot_joint()

# %% DO ICA
n_comp = 15 #number of components
ica = ICA(n_components=n_comp, max_iter="auto", random_state=97)
# fitting ICA into raw
ica.fit(eeg_raw)
# Plot ICA
ica.plot_sources(eeg_raw, show_scrollbars=True)
ica.plot_components()
# See each components' explainable variance
for comp in range(n_comp):
    explained_var_ratio = ica.get_explained_variance_ratio(eeg_raw, components=[comp], ch_type="eeg")
    # This time, print as percentage.
    ratio_percent = round(100 * explained_var_ratio["eeg"])
    print(
        f"Fraction of variance in EEG signal explained by first component: "
        f"{ratio_percent}%"
    )

# Exclude eye movement ICA components
ica.exclude=[1,3]
# %% Plot EEG after removing eye movement
ica.plot_overlay(eeg_raw, picks="eeg", start=2000, stop=4000)
eeg_raw = ica.apply(eeg_raw)
eeg_raw.plot()
# Save ICA
ica.save(data_root+'/'+subject+'_ICA.fif', overwrite=True)

# %% Loop subject see ICA
for subject in subject_list:
    if subject == 'chimera_012' or subject =='chimera_019' or subject =='chimera_021':
        continue
    else:
        print(subject)
        data_root = exp_path+'subject'+subject[7:]
        ica = mne.preprocessing.read_ica(data_root+'/'+subject+'_ICA.fif')
        print(ica.exclude)
