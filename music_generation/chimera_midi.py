#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Making Chimera music

Created on Mon Mar 13 13:47:32 2023

@author: tshan@urmc-sh.rochester.edu
"""

import numpy as np
import mido
import pygame.mixer
import pandas as pd
import random
from midi2audio import FluidSynth
from scipy.io import wavfile
import os

# %%
def chimera(pitch_mid, rhythm_mid):
    """
    Parameters
    ----------
    pitch_mid : midi file as pitch, read by MIDO.
    rhythm_mid : midi file as rhythm, read by MIDO.

    Returns
    -------
    mid_new : the Chimera music.

    """
    pitch_note_track = pitch_mid.tracks[1]
    rhythm_note_track = rhythm_mid.tracks[1]
    
    # For pitch mid
    pitch_on_off_list = []
    pitch_note_list = []
    pitch_velocity_list = []
    pitch_time_list = []
    for i, msg in enumerate(pitch_note_track):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
            pitch_on_off_list += [msg_str_list[0]]
            pitch_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
            pitch_velocity_list += [int(msg_str_list[3].split('=')[1])] # extract velocity
            pitch_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
        else:
            continue
    pitch_note_true_list_index = [i for i in range(len(pitch_on_off_list)) if pitch_on_off_list[i] == "note_on"]
    pitch_note_true_list = [pitch_note_list[i] for i in pitch_note_true_list_index]
    
    # For rhythm mid
    rhythm_on_off_list = []
    rhythm_note_list = []
    rhythm_velocity_list = []
    rhythm_time_list = []
    for i, msg in enumerate(rhythm_note_track):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
            rhythm_on_off_list += [msg_str_list[0]]
            rhythm_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
            rhythm_velocity_list += [int(msg_str_list[3].split('=')[1])] # extract velocity
            rhythm_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
        else:
            continue
    rhythm_on_index = [i for i in range(len(rhythm_on_off_list)) if rhythm_on_off_list[i] == "note_on"]
    rhythm_off_index = [i for i in range(len(rhythm_on_off_list)) if rhythm_on_off_list[i] == "note_off"]
    
    # Compare pitch midi note list and rhythm midi note list, cut the larger one
    if len(pitch_note_true_list) >= len(rhythm_off_index):
        pitch_note_true_list = pitch_note_true_list[:len(rhythm_off_index)]
    elif len(pitch_note_true_list) < len(rhythm_off_index):
        rhythm_note_list = rhythm_note_list[:len(pitch_note_list)]
        rhythm_on_off_list = rhythm_on_off_list[:len(pitch_note_list)]
        rhythm_velocity_list = rhythm_velocity_list[:len(pitch_note_list)]
        rhythm_time_list = rhythm_time_list[:len(pitch_note_list)]
            
    # Create new midi
    mid_new = mido.MidiFile()
    mid_new.ticks_per_beat = rhythm_mid.ticks_per_beat
    track = mido.MidiTrack()
    mid_new.tracks.append(rhythm_mid.tracks[0]) # pass the meta data from original file
    mid_new.tracks.append(track)
    for i , msg in enumerate(rhythm_mid.tracks[1]):
        print(msg)
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] != 'note_on':
            track.append(msg) # pass the track meta data
        else:
            break
    for o, n, v, t in zip(rhythm_on_off_list, pitch_note_list, rhythm_velocity_list, rhythm_time_list):
        track.append(mido.Message(o, channel=0, note=n, velocity=v, time=t))
    track.append(mido.MetaMessage('end_of_track', time=0))
    # Avoid last "note_on" has velocity
    for i, msg in enumerate(track):
        if (i==(len(track)-2)) & (msg.type=="note_on"):
            msg.velocity = 0
    return mid_new


def change_instrument(mid, instrument):
    """
    Parameters
    ----------
    mid : midi file read by MIDO.
    instrument : MIDI program instrument, see
        https://jazz-soft.net/demo/GeneralMidi.html

    Returns
    -------
    mid_new : new midi file.

    """
    mid_new = mido.MidiFile()
    mid_new.ticks_per_beat = mid.ticks_per_beat
    track = mido.MidiTrack()
    mid_new.tracks.append(mid.tracks[0]) # pass the meta data from original file
    mid_new.tracks.append(track)
    for i , msg in enumerate(mid.tracks[1]):
        if msg.type == "program_change":
            track.append(mido.Message('program_change', channel=1, program=instrument, time=0))
        else:
            track.append(msg)
    return mid_new

def separate_note(mid_path):
    mid = mido.MidiFile(mid_path)
    # Odd note
    mid_odd = mido.MidiFile()
    mid_odd.ticks_per_beat = mid.ticks_per_beat
    track_odd = mido.MidiTrack()
    mid_odd.tracks.append(mid.tracks[0]) # pass the meta data from original file
    mid_odd.tracks.append(track_odd)
    for i , msg in enumerate(mid.tracks[1]):
        if i < 2 or i==(len(mid.tracks[1])-1):
            track_odd.append(msg)
        else:
            if ((i-2)%4 in [0,1]) or ((i==(len(mid.tracks[1])-2)) & (msg.type=='note_on')):
                msg.velocity = 0
                track_odd.append(msg)
            else:
                track_odd.append(msg)
                
    mid = mido.MidiFile(mid_path)
    # Even note
    mid_even = mido.MidiFile()
    mid_even.ticks_per_beat = mid.ticks_per_beat
    track_even = mido.MidiTrack()
    mid_even.tracks.append(mid.tracks[0]) # pass the meta data from original file
    mid_even.tracks.append(track_even)
    for i , msg in enumerate(mid.tracks[1]):
        if i < 2 or i==(len(mid.tracks[1])-1):
            track_even.append(msg)
        else:
            if ((i-2)%4 in [2,3]) or ((i==(len(mid.tracks[1])-2)) & (msg.type=='note_on')):
                msg.velocity = 0
                track_even.append(msg)
            else:
                track_even.append(msg)
    return mid_odd, mid_even
    
# %% Folks Note duration profile
import os
import shutil

midi_path_folk = '/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/'
datasheet_all = pd.DataFrame()

for filename in os.listdir(midi_path_folk+'europa-midi-all/'):
    ID = filename[:-4]
    f = os.path.join(midi_path_folk+'europa-midi-all/', filename)
    print("processing... " + ID)
    
    # Find time signature
    mid = mido.MidiFile(f)
    for msg in mid.tracks[0]:
        if msg.dict()['type'] == 'time_signature':
            numerator = str(msg.dict()['numerator'])
            denominator = str(msg.dict()['denominator'])
        else:
            continue
    length = mid.length

    # Midi file msg dataframe
    # Extract information
    on_off_list = []
    note_list = []
    velocity_list = []
    time_list = []
    for i, msg in enumerate(mid.tracks[1]):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
            on_off_list += [msg_str_list[0]]
            note_list += [int(msg_str_list[2].split('=')[1])] # extract note
            velocity_list += [int(msg_str_list[3].split('=')[1])] # extract velocity
            time_list += [int(msg_str_list[4].split('=')[1])] # extract time
        else:
            continue
    time_real_list = time_list.copy()
    for li in range(1, len(time_list)):
        time_real_list[li] = time_real_list[li-1] + time_list[li]
    # Delete the last no action note
    if velocity_list[-1] == 0 and on_off_list[-1] == "note_on":
        on_off_list = on_off_list[:-1]
        note_list = note_list[:-1]
        velocity_list = velocity_list[:-1]
        time_list = time_list[:-1]
        time_real_list = time_real_list[:-1]
    # Make dataframe
    temp_midi_df = pd.DataFrame()
    temp_midi_df['note'] = note_list
    temp_midi_df['type'] = on_off_list
    temp_midi_df['time'] = time_list
    temp_midi_df['time_real']  = time_real_list
    temp_midi_df['velocity'] = velocity_list
    # Make note on/off time and duration 
    note_dur = []
    for i, row in temp_midi_df.iterrows():
        if row["type"] == "note_on":
            same_note_list = list(temp_midi_df[temp_midi_df['note']==row["note"]].index)
            same_note_list = [n for n in same_note_list if n > float(i)]
            next_ind = same_note_list[np.asarray([n-i for n in same_note_list]).argmin()]
            
            note_dur += [float(temp_midi_df.loc[next_ind, ["time_real"]]) - float(temp_midi_df.loc[i, ["time_real"]])]
        else:
            continue
    # Compte mean duration
    mean_note_dur = np.average(note_dur)
    num_of_note = len(note_list)/2
    temp_df = {"ID": ID, "numerator": numerator, "denominator": denominator,
               "mean_note_dur": mean_note_dur, "length": length,
               "num_of_note":num_of_note}
    datasheet_all = datasheet_all.append(temp_df, ignore_index=True)
# Remove duplicated piece
datasheet_deduped = datasheet_all.drop_duplicates(subset=datasheet_all.columns.difference(['ID']))
 
# Save new profile data
datasheet_deduped.to_csv('/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/datasheet_meta.csv')

# Move deduped files to a new folder
for i in datasheet_deduped['ID']:
    src = midi_path_folk+'europa-midi-all/' + i +'.mid'
    dst = midi_path_folk+'europa-midi-train/' + i + '.mid'
    shutil.copy(src, dst)

# select data with length > 25, mean_note_dur < 70
datasheet_select = datasheet_deduped[(datasheet_deduped["length"] > 30) & (datasheet_deduped["mean_note_dur"] < 80)]
# After hearing, delet ones have less pattern
ID_rm = ['deut0705','oberhas1','deut1855','luxemb07']
datasheet_select_refine = datasheet_select[~datasheet_select['ID'].isin(ID_rm)]

# Devide in two set with different signatures
datasheet_3 = datasheet_select_refine[datasheet_select_refine['numerator'].isin(["3","6"])]
datasheet_2 = datasheet_select_refine[~datasheet_select_refine['numerator'].isin(["3","6"])]
# Sort and match by number of note
datasheet_3.sort_values(by=['num_of_note'])
datasheet_2.sort_values(by=['num_of_note'])

pairs = [["deut256", "deut2864"], ["deut1824","elsass54"],["deut4471","deut1786"],
         ["deut2880","deut1782"],["deut075","deut2641"],["deut3767","deut296"],
         ["deut2559","deut377"],["deut2565","ussr27"],["suisse01","lothr004"],
         ["deut010","deut4726"],["deut5078","deut2876"],["deut4995","deut330"],
         ["deut4479","deut2767"],["tirol10","deut2575"],["deut0827","deut488"],
         ["oestr003","deut1792"],["deut303","deut1687"],["deut3768","deut2666"],
         ["deut4576","deut0706"],["deut5075","deut1996"],["deut1902","deut2035"],
         ["deut040","oestr033"],["deut4813","deut2904"],["suisse23","deut305"],
         ["deut4814","deut0704"],["steier09","deut3757"],["oestr023","oestr098"],
         ["siebethl","deut4838"],["deut2788","deut4461"],["deut2744","deut4484"],
         ["neder062","deut2881"],["deut446","deut494"],["deut051","tirol04"]]

pairs = np.asarray(pairs)

for pi in range(len(pairs)):
    # Read pairs path
    mid_0_path = midi_path_folk + 'europa-midi-train/' + pairs[pi][0] + '.mid'
    mid_1_path = midi_path_folk + 'europa-midi-train/' + pairs[pi][1] + '.mid'
    # Read midi file
    mid_0 = mido.MidiFile(mid_0_path)
    mid_1 = mido.MidiFile(mid_1_path)
    # Make Chimeras
    mid_new_0 = chimera(mid_0, mid_1)
    mid_new_1 = chimera(mid_1, mid_0)
    # Save chimeras
    mid_new_0.save(midi_path_folk+'europa-stim-chimera/'+'chimera_'+pairs[pi][0]+
                   '_'+pairs[pi][1]+'.mid')
    mid_new_1.save(midi_path_folk+'europa-stim-chimera/'+'chimera_'+pairs[pi][1]+
                   '_'+pairs[pi][0]+'.mid')
    # Move Originals
    os.rename(mid_0_path, midi_path_folk+'europa-stim-ori/'+pairs[pi][0]+'.mid')
    os.rename(mid_1_path, midi_path_folk+'europa-stim-ori/'+pairs[pi][1]+'.mid')

chimera_length = 0
for filename in os.listdir(midi_path_folk+'europa-stim-chimera/'):
    ID = filename[:-4]
    f = os.path.join(midi_path_folk+'europa-stim-chimera/', filename)
    print("processing... " + ID)

    mid = mido.MidiFile(f)
    chimera_length += mid.length
    # Change instrument
    mid_new = change_instrument(mid, 0)
    mid_new.save(f)

ori_length = 0
for filename in os.listdir(midi_path_folk+'europa-stim-ori/'):
    ID = filename[:-4]
    f = os.path.join(midi_path_folk+'europa-stim-ori/', filename)
    print("processing... " + ID)

    mid = mido.MidiFile(f)
    ori_length += mid.length
    # Change instrument
    mid_new = change_instrument(mid, 0)
    mid_new.save(f)

# %% Render MIDI file to wav

mid_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-stim-chimera/"
wav_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-stim-wav/"
mid_sep_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-midi-chimera-sep/"
wav_sep_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-wav-chimera-sep/"
ref_rms = 0.01
for filename in os.listdir(mid_folder):
    f = os.path.join(mid_folder, filename)
    mid_odd, mid_even = separate_note(f)
    mid_odd.save(mid_sep_folder+filename[:-4]+"_odd.mid")
    mid_even.save(mid_sep_folder+filename[:-4]+"_even.mid")
    # synth wave
    fs = FluidSynth(sound_font='/usr/share/sounds/sf2/FluidR3_GM.sf2')
    fs = FluidSynth(sample_rate=48000)
    fs.midi_to_audio(mid_sep_folder+filename[:-4]+"_odd.mid", wav_sep_folder+filename[:-4]+"_odd.wav")
    fs.midi_to_audio(mid_sep_folder+filename[:-4]+"_even.mid", wav_sep_folder+filename[:-4]+"_even.wav")
    freq, wave_odd = wavfile.read(wav_sep_folder+filename[:-4]+"_odd.wav")
    wave_odd = np.mean(wave_odd, axis=1)
    freq, wave_even = wavfile.read(wav_sep_folder+filename[:-4]+"_even.wav")
    wave_even = np.mean(wave_even, axis=1)
    # flip the polarity of even note
    wave_new = wave_odd + (-wave_even)
    # normalize the waveform
    wave_norm = wave_new / np.std(wave_new) * ref_rms
    # save the file
    wavfile.write(wav_folder+filename[:-4]+".wav", freq, wave_norm)

mid = mido.MidiFile("/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-stim-chimera/chimera_deut1786_deut4471.mid")
mid.tracks[1][-10:]
mid_2 = mido.MidiFile("/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-midi-ori-sep/deut1786_odd.mid")
mid_2.tracks[1][-10:]


# %% Fix two midi files
chimera_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-stim-chimera/"
ori_folder = "/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/folk_corpus/europa-stim-ori/"
file_1 = "chimera_deut3768_deut2666.mid"
file_2 = "chimera_deut2666_deut3768.mid"
ori_file_1 = "deut3768.mid"
ori_file_2 = "deut2666.mid"
mid_1 = mido.MidiFile(chimera_folder+file_1)
mid_2 = mido.MidiFile(chimera_folder+file_2)
ori_1 = mido.MidiFile(ori_folder+ori_file_1)
ori_2 = mido.MidiFile(ori_folder+ori_file_2)

# For pitch mid
mid_1_note_list = []
mid_1_time_list = []

for i, msg in enumerate(mid_1.tracks[1]):
    msg_str_list = str(msg).split(' ')
    if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
        mid_1_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
        mid_1_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
    else:
        continue

ori_1_on_off_list = []
ori_1_note_list = []
ori_1_time_list = []
for i, msg in enumerate(ori_1.tracks[1]):
    msg_str_list = str(msg).split(' ')
    if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
        ori_1_on_off_list += [msg_str_list[0]]
        ori_1_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
        ori_1_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
    else:
        continue

tf = []
for mid_1_i, ori_1_i in zip(mid_1_note_list, ori_1_note_list[:224]):
    if mid_1_i == ori_1_i:
        tf += [True]
    else:
        tf += [False]
        
ori_2_note_list = []
ori_2_time_list = []
for i, msg in enumerate(ori_2.tracks[1]):
    msg_str_list = str(msg).split(' ')
    if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
        ori_2_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
        ori_2_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
    else:
        continue

tf = []
for mid_1_i, ori_2_i in zip(mid_1_time_list, ori_2_time_list[:225]):
    if mid_1_i == ori_2_i:
        tf += [True]
    else:
        tf += [False]
        

mid_2_note_list = []
mid_2_time_list = []

for i, msg in enumerate(mid_2.tracks[1]):
    msg_str_list = str(msg).split(' ')
    if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
        mid_2_note_list += [int(msg_str_list[2].split('=')[1])] # extract note
        mid_2_time_list += [int(msg_str_list[4].split('=')[1])] # extract time
    else:
        continue

tf = []
for mid_2_i, ori_2_i in zip(mid_2_note_list, ori_2_note_list[:222]):
    if mid_2_i == ori_2_i:
        tf += [True]
    else:
        tf += [False]
        
tf = []
for mid_2_i, ori_1_i in zip(mid_2_time_list, ori_1_time_list[:222]):
    if mid_2_i == ori_1_i:
        tf += [True]
    else:
        tf += [False]


# Create new midi
mid_new = mido.MidiFile()
mid_new.ticks_per_beat = ori_1.ticks_per_beat
track = mido.MidiTrack()
mid_new.tracks.append(ori_1.tracks[0]) # pass the meta data from original file
mid_new.tracks.append(track)
for i , msg in enumerate(ori_1.tracks[1]):
    print(msg)
    msg_str_list = str(msg).split(' ')
    if msg_str_list[0] != 'note_on':
        track.append(msg) # pass the track meta data
    else:
        break
for o, n, t in zip(ori_1_on_off_list, ori_2_note_list, ori_1_time_list):
    track.append(mido.Message(o, channel=0, note=n, velocity=64, time=t))
track.append(mido.MetaMessage('end_of_track', time=0))
# Avoid last "note_on" has velocity
for i, msg in enumerate(track):
    if (i==(len(track)-2)) & (msg.type=="note_on"):
        msg.velocity = 0

mid_new.save("chimera_deut2666_deut3768.mid")

f="chimera_deut3768_deut2666.mid"
mid_odd, mid_even = separate_note(f)
mid_odd.save(mid_sep_folder+f[:-4]+"_odd.mid")
mid_even.save(mid_sep_folder+f[:-4]+"_even.mid")
# synth wave
fs = FluidSynth(sound_font='/usr/share/sounds/sf2/FluidR3_GM.sf2')
fs = FluidSynth(sample_rate=48000)
fs.midi_to_audio(mid_sep_folder+f[:-4]+"_odd.mid", wav_sep_folder+f[:-4]+"_odd.wav")
fs.midi_to_audio(mid_sep_folder+f[:-4]+"_even.mid", wav_sep_folder+f[:-4]+"_even.wav")
freq, wave_odd = wavfile.read(wav_sep_folder+f[:-4]+"_odd.wav")
wave_odd = np.mean(wave_odd, axis=1)
freq, wave_even = wavfile.read(wav_sep_folder+f[:-4]+"_even.wav")
wave_even = np.mean(wave_even, axis=1)
# flip the polarity of even note
wave_new = wave_odd + (-wave_even)
# normalize the waveform
wave_norm = wave_new / np.std(wave_new) * ref_rms
# save the file
wavfile.write(wav_folder+f[:-4]+".wav", freq, wave_norm)
