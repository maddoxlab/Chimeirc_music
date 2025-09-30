#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 11:37:31 2023

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

chimera_path='/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/europa-stim-chimera'
chimera_files = os.listdir(chimera_path)
for f in chimera_files:
    if not f.endswith('mid'):
        chimera_files.remove(f)
chimera_files = [f.split('.')[0] for f in chimera_files]
n_epoch_chimera=len(chimera_files)

ori_path='/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/europa-stim-ori'
ori_files = os.listdir(ori_path)
for f in ori_files:
    if not f.endswith('mid'):
        ori_files.remove(f)
ori_files = [f.split('.')[0] for f in ori_files]
# Pairs
hash_chimera = []
pairs_ori = []
pairs_chimera = []
for i in range(n_epoch_chimera):
    if i in hash_chimera:
        continue
    else:
        file1 = chimera_files[i].split("_")[1]
        file2 = chimera_files[i].split("_")[2]
        ind1 = ori_files.index(file1)
        ind2 = ori_files.index(file2)
        pairs_ori += [[ind1,ind2]]
        for k in range(i+1, n_epoch_chimera):
            if chimera_files[k].startswith("chimera_"+file2+"_"):
                hash_chimera += [i]
                hash_chimera += [k]
                pairs_chimera += [[i,k]]

# make key list
dic = {}
for i in range(33):
    # dic[i] = [ori_files[pairs_ori[i][0]], ori_files[pairs_ori[i][1]],
    #           chimera_files[pairs_chimera[i][0]], chimera_files[pairs_chimera[i][1]]]
    dic[ori_files[pairs_ori[i][0]]] = i
    dic[ori_files[pairs_ori[i][1]]] = i
    dic[chimera_files[pairs_chimera[i][0]]] = i
    dic[chimera_files[pairs_chimera[i][1]]] = i

write_hdf5('/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/MusicExp/folk_corpus/regressor/set_pairs.hdf5', dic, overwrite=True)
