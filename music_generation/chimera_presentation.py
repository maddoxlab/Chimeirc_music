#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 11:29:42 2023

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from expyfun import ExperimentController, decimals_to_binary
from expyfun.stimuli import window_edges
from expyfun.visual import ProgressBar
from expyfun.io import read_wav
import os
import random
from expyfun.visual import Circle
from datetime import datetime

#%% Set parameters
n_epoch_total = 132
n_epoch_ori = 66
n_epoch_chimera = 66

# hardware
fs = 48000
n_channel = 2
rms = 0.01
stim_db = 65 #????

pause_dur = 1  # pause duration between each epoch

# %%
exp_path = "C://Users/labuser/Code/Chimera_Music/"
file_path = exp_path+"europa-stim-wav/"

stim_type_list = []
stim_name_list = []
stim_path_list = []
stim_dur_list = []
for filename in os.listdir(file_path):
    stim_name_list += [filename]
    stim_path_list += [os.path.join(file_path, filename)]
    # Type index: 0 - original, 1 - chimera
    if filename.startswith("chimera"):
        stim_type_list += [1]
    else:
        stim_type_list += [0]
    temp, fs = read_wav(os.path.join(file_path, filename))
    stim_dur_list += [len(temp[0])/fs]
stim_df = pd.DataFrame(list(zip(stim_name_list,stim_path_list,stim_type_list,stim_dur_list)),
                       columns=['name','path','type','duration'])

stim_df_shuff = stim_df.sample(frac=1).reset_index(drop=True)

# %%
instruction = (' Hi, thank you for participating this study!'
              '\n In this experiment, you will hear music clips, each around 30 seconds.'
              '\n After each music, you will be asked "how do you like the music?"'
              '\n Please use your mouse to choose from a scale of 1 to 7 according to your personal preference.'
              '\n Press the SPACE bar to continue.')
pause_trial = [44, 88]
frac = ['1/3','2/3']
break_dur = 30
# %% Experiment
n_bits_epoch = int(np.ceil(np.log2(n_epoch_total)))
n_bits_type = int(np.ceil(np.log2(2)))

ec_args = dict(exp_name='Chimera Music', window_size=(1920, 1080),
               session='00', full_screen=True, n_channels=n_channel,
               version='dev', enable_video=True, stim_fs=fs, stim_db=60,
               force_quit=['end'])
trial_start_time = -np.inf
# %%
with ExperimentController(**ec_args) as ec:
    ec.screen_prompt(instruction, live_keys=['space'])
    ec.screen_prompt(' There will be two short breaks during the experiment.'
                      '\n Press the SPACE bar to begin.', live_keys=['space'])
    pb = ProgressBar(ec, [0, -.2, 1.5, .2], colors=('xkcd:teal blue', 'w'))
    for ei in range(n_epoch_total):
        if ei in pause_trial:  # shows pb at 1/3, 2/3 complete
            # calculate percent done and update the bar object
            percent = ei / n_epoch_total * 100
            pb.update_bar(percent)
            # display the progress bar with some text
            ec.screen_text('You\'ve completed {}. Take a break! (30 seconds '
                            'minimum).'.format(frac[pause_trial.index(ei)]), [0, .1], wrap=False)
            pb.draw()
            ec.flip()
            # subject uses any key press to proceed
            ec.wait_secs(break_dur)
            ec.screen_prompt('Press space to continue.', live_keys=['space'],
                              max_wait=np.inf)
        # Draw progress bar
        ec.screen_text('Trial number ' + str(ei+1) + ' out of ' + str(int(n_epoch_total)) + ' trials.')
        pb.draw()
        # Load trial parameters
        trial_id = stim_df_shuff.loc[ei, 'name']
        temp, fs = read_wav(stim_df_shuff.loc[ei, 'path'])
        trial_dur = stim_df_shuff.loc[ei, 'duration']
        type_number = stim_df_shuff.loc[ei, 'type']
        # Load buffer, define trial
        ec.load_buffer(temp)
        ec.identify_trial(ec_id=trial_id, ttl_id=[])
        ec.wait_until(trial_start_time + trial_dur + pause_dur)
        # Start stimulus
        trial_start_time = ec.start_stimulus()
        ec.wait_secs(0.1)
        # Trigger
        ec.stamp_triggers([(b + 1) * 4 for b in decimals_to_binary([type_number, ei], [n_bits_type, n_bits_epoch])])
        while ec.current_time < trial_start_time + trial_dur:
            ec.check_force_quit()
            ec.wait_secs(0.1)
        ec.stop()
        #Ask prefenece
        ec.toggle_cursor(False)
        ec.wait_secs(0.1)
        ec.screen_text("How much do you like the music?", pos=[0, 0.2], color='w')
        ec.screen_text("1 = don't like it at all", pos=[0, 0], color='w')
        ec.screen_text("7 = like it very much", pos=[1.25, 0], color='w')
        init_circles = []
        for i in range(7):
            init_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.25), -0.25), units='norm',
                                    fill_color=None, line_color='white', line_width=5)]
            ec.screen_text(str(i+1), pos=[(i)*0.25+0.04, -0.15],units='norm', color='w')
        for c in init_circles:
            c.draw()
        ec.flip()
        click, ind = ec.wait_for_click_on(init_circles, max_wait=np.inf)
        after_circles = []
        for i in range(7):
            if i == ind:
                after_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.25), -0.25), units='norm',
                                         fill_color='white', line_color='white', line_width=5)]
            else:
                after_circles += [Circle(ec, radius=(0.02, 0.03), pos=((-1+(i+1)*0.25), -0.25), units='norm',
                                        fill_color=None, line_color='white', line_width=5)]
            ec.screen_text(str(i+1), pos=[(i)*0.25+0.04, -0.15],units='norm', color='w')
        for c in after_circles:
            c.draw()
        ec.flip()
        # Save data
        ec.write_data_line("Trial Params", dict(trial_num=ei,
                                                trial_id=trial_id,
                                                type_number=type_number,
                                                preferenc=ind))
        ec.trial_ok()
        # Update progress bar
        pb.update_bar((ei + 1) / n_epoch_total * 100)
        pb.draw()
    # End exp
    ec.screen_prompt('ya did it!')
True
# Save the stim_df_shuffle
now = datetime.now()
stim_df_shuff.to_csv(exp_path+"data/"+str(now.date())+'-'+str(now.date())[:2]+"_stim_dataframe.csv")
