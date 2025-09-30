import numpy as np
import mido
import pygame.mixer
import pandas as pd
import random

# %% Functions for Bach

def msg2dict(mid):
    note_track = mid.tracks[1]
    note_list = []
    velocity_list = []
    time_list = []
    for i, msg in enumerate(note_track):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] == 'note_on':
            note_list += [int(msg_str_list[2].split('=')[1])] # extract note
            velocity_list += [int(msg_str_list[3].split('=')[1])] # extract velocity
            time_list += [int(msg_str_list[4].split('=')[1])] # extract time
        else:
            continue
    note_true_list = note_list[::2]
    duration_true_list = time_list[1::2]
    return dict(note_list=note_list, velocity_list=velocity_list, time_list=time_list,
                note_true_list=note_true_list, duration_true_list=duration_true_list)

def dict2msg(msg_dict, old_mid, mode, true_list):
    if mode=='pitch': 
        note_shuffled_list = []
        for i in range(len(true_list)):
            note_shuffled_list += [true_list[i]]*2
        velocity_list = msg_dict['velocity_list']
        time_list = msg_dict['time_list']
        # new midi file and tracks
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(old_mid.tracks[0]) # pass the meta data from original file
        mid_new.tracks.append(track)
        for i , msg in enumerate(old_mid.tracks[1]):
            msg_str_list = str(msg).split(' ')
            if msg_str_list[0] != 'note_on':
                track.append(msg) # pass the track meta data
            else:
                break
        for n, v, t in zip(note_shuffled_list, velocity_list, time_list):
            track.append(mido.Message('note_on', channel=0, note=n, velocity=v, time=t))
    elif mode=='duration':
        duration_shuffled_list = msg_dict['time_list']
        for i in range(len(true_list)):
            duration_shuffled_list[i*2+1] = true_list[i]
        note_list = msg_dict['note_list']
        velocity_list = msg_dict['velocity_list']
        # new midi file and tracks
        mid_new = mido.MidiFile()
        track = mido.MidiTrack()
        mid_new.tracks.append(old_mid.tracks[0]) # pass the meta data from original file
        mid_new.tracks.append(track)
        for i , msg in enumerate(old_mid.tracks[1]):
            msg_str_list = str(msg).split(' ')
            if msg_str_list[0] != 'note_on':
                track.append(msg) # pass the track meta data
            else:
                break
        for n, v, t in zip(note_list, velocity_list, duration_shuffled_list):
            track.append(mido.Message('note_on', channel=0, note=n, velocity=v, time=t))
    return mid_new

# %% Load 1 Midi file
midi_path = './midi_files/'
pygame.mixer.init()
pygame.mixer.music.load(midi_path+'fp-4bou.mid')
pygame.mixer.music.play()

mid = mido.MidiFile(midi_path+'/fp-4bou.mid')
for i, track in enumerate(mid.tracks):
    for msg in track:
        print(msg)

for msg in mid.tracks[1][:100]:
    print(msg)

msg_dict = msg2dict(mid)

# Pitch shuffling
note_shuffled_true_list = random.sample(msg_dict['note_true_list'], len(msg_dict['note_true_list']))

mid_new = dict2msg(msg_dict, note_shuffled_true_list,mid)

mid_new.save('mid_new.mid')
pygame.mixer.music.load('mid_new.mid')
pygame.mixer.music.play()

# Time shuffling
duration_shuffled_true_list = random.sample(msg_dict['duration_true_list'],
                                            len(msg_dict['duration_true_list']))
mid_new = dict2msg(msg_dict, mid, 'duration', duration_shuffled_true_list)
mid_new.save('mid_new_duration.mid')
pygame.mixer.music.load('mid_new_duration.mid')
pygame.mixer.music.play()

# %% Functions for MCCC
def shuffle(input_list, count):
    '''Shuffles any n number of values in a list'''
    indices_to_shuffle = random.sample(range(len(input_list)), k=count)
    to_shuffle = [input_list[i] for i in indices_to_shuffle]
    random.shuffle(to_shuffle)
    for index, value in enumerate(to_shuffle):
        old_index = indices_to_shuffle[index]
        input_list[old_index] = value
    return input_list

def msg2dict(mid):
    note_track = mid.tracks[1]
    on_off_list = []
    note_list = []
    velocity_list = []
    time_list = []
    for i, msg in enumerate(note_track):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] == 'note_on' or msg_str_list[0]== 'note_off':
            on_off_list += [msg_str_list[0]]
            note_list += [int(msg_str_list[2].split('=')[1])] # extract note
            velocity_list += [int(msg_str_list[3].split('=')[1])] # extract velocity
            time_list += [int(msg_str_list[4].split('=')[1])] # extract time
        else:
            continue
    note_true_list = note_list[::2]
    if len(note_list)%2 != 0:
        note_true_list = note_true_list[:-1]
    duration_true_list = time_list[1::2]
    return dict(on_off_list=on_off_list,note_list=note_list, 
                velocity_list=velocity_list, time_list=time_list,
                note_true_list=note_true_list, 
                duration_true_list=duration_true_list)

def dict2msg(msg_dict, old_mid, mode, level):
    count = int(np.floor(len(msg_dict['note_true_list']) * level*0.01))
    note_shuffled_true_list = shuffle(msg_dict['note_true_list'], count)
    note_shuffled_list = []
    for i in range(len(note_shuffled_true_list)):
        note_shuffled_list += [note_shuffled_true_list[i]]*2
    duration_shuffled_true_list = shuffle(msg_dict['duration_true_list'], count)
    duration_shuffled_list = msg_dict['time_list']
    for i in range(len(duration_shuffled_true_list)):
        duration_shuffled_list[i*2+1] = duration_shuffled_true_list[i]
    on_off_list = msg_dict['on_off_list']
    note_list = msg_dict['note_list']
    velocity_list = msg_dict['velocity_list']
    time_list = msg_dict['time_list']
    # new midi file and tracks
    mid_new = mido.MidiFile()
    mid_new.ticks_per_beat = old_mid.ticks_per_beat
    track = mido.MidiTrack()
    mid_new.tracks.append(old_mid.tracks[0]) # pass the meta data from original file
    mid_new.tracks.append(track)
    for i , msg in enumerate(old_mid.tracks[1]):
        msg_str_list = str(msg).split(' ')
        if msg_str_list[0] != 'note_on':
            track.append(msg) # pass the track meta data
        else:
            break
    # Choose mode
    if mode=='pitch': 
        for o, n, v, t in zip(on_off_list, note_shuffled_list, velocity_list, time_list):
            track.append(mido.Message(o, channel=0, note=n, velocity=v, time=t))
    elif mode=='duration':
        for o, n, v, t in zip(on_off_list, note_list, velocity_list, duration_shuffled_list):
            track.append(mido.Message(o, channel=0, note=n, velocity=v, time=t))
    elif mode=='both':
        for o, n, v, t in zip(on_off_list, note_shuffled_list, velocity_list, duration_shuffled_list):
            track.append(mido.Message(o, channel=0, note=n, velocity=v, time=t))
    else:
        raise ValueError('Method should be either "pitch" or "duration" or "both".')
    return mid_new

def mid_scramble(mid, mode, level):
    msg_dict = msg2dict(mid)
    mid_new = dict2msg(msg_dict, mid, mode, level)
    return mid_new
# %% 
# Pitch shuffling diff level
midi_path = '/home/urmc-sh.rochester.edu/tshan/AMPLab/MusicExp/midi_files/osfstorage-archive/MCCC-midi/'
mid = mido.MidiFile(midi_path+'ILB0165_04.mid')
levels = [5, 10, 20, 40, 80, 100]
for lv in levels:
    mid_pitch_5 = mid_scramble(mid, 'pitch', lv)
    mid_pitch_5.save('ILB0165_04_pitch_{}.mid'.format(lv))
    
pygame.mixer.init()
pygame.mixer.music.load('ILB0165_04_pitch_40.mid')
pygame.mixer.music.play()


# Time shuffling 
levels = [5, 10, 20, 40, 80, 100]
for lv in levels:
    mid_pitch_5 = mid_scramble(mid, 'duration', lv)
    mid_pitch_5.save('ILB0165_04_duration_{}.mid'.format(lv))
    
pygame.mixer.init()
pygame.mixer.music.load('ILB0165_04_duration_40.mid')
pygame.mixer.music.play()

# Both shuffling
levels = [5, 10, 20, 40, 80, 100]
for lv in levels:
    mid_pitch_5 = mid_scramble(mid, 'both', lv)
    mid_pitch_5.save('ILB0165_04_both_{}.mid'.format(lv))

pygame.mixer.init()
pygame.mixer.music.load('ILB0165_04_both_5.mid')
pygame.mixer.music.play()

