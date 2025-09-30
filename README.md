# Chimeric music reveals an interaction of pitch and time in electrophysiological signatures of music encoding

This repository contains code for paper [Chimeric music reveals an interaction of pitch and time in electrophysiological signatures of music encoding](https://doi.org/10.1101/2024.11.04.621689)

The data used in this study can be found on [OpenNeuro.org](https://openneuro.org/datasets/ds006735).
## Repository Organization

The code has been organized into the following functional categories:

### eeg_analysis/
**EEG Signal Processing & Analysis**
- `ICA_eyemovement.py` - Independent Component Analysis for eye movement artifact removal
- `derive_ERP_ICA.py` - Event-Related Potential analysis with ICA preprocessing
- `derive_ERP_ICA_downbeat.py` - ERP analysis focused on downbeat processing

### music_generation/
**Music Stimulus Generation**
- `chimera_midi.py` - Generate Chimera music by combining pitch and rhythm from different sources
- `chimera_presentation.py` - Present Chimera music stimuli in experiments
- `midi_scramble.py` - MIDI file scrambling utilities
- `original_chimera_pairing.py` - Pair original and Chimera music stimuli

### trf_analysis/
**Temporal Response Function (TRF) Analysis**
- `derive_TRF_ICA_downbeat.py` - TRF analysis focused on downbeat processing
- `derive_TRF_ICA_both_pitch-time.py` - TRF analysis for pitch-time interactions
- `derive_TRF_ICA_LTM-STM.py` - Long-term vs Short-term memory TRF analysis
- `TRF_analysis_all_LTM_STM.py` - LTM-STM comparison analysis
- `TRF_analysis_all_both_pitch-time.py` - Pitch-time interaction analysis
- `TRF_find_lambda_mode.py` - Lambda parameter optimization using mode selection
- `TRF_find_lambda_new_algorithm.py` - Advanced lambda parameter optimization

### abr_analysis/
**Auditory Brainstem Response (ABR) Analysis**
- `derive_ABR_new.py` - Updated ABR analysis with improved processing

### statistics/
**Statistical Analysis & Visualization**
- `preference_analysis.py` - Analyze participant music preferences
- `show_plot_publication.py` - Generate publication-quality plots
- `note_timing_dist.py` - Analyze note timing distributions

### revision_edit/
**Revision edits**
- `Jneuro_edit_review.ipynb` - revision edit analysis
- `Jneuro_edit_review_plot.ipynb` - Plotting for revision

### MusicianshipQuestionnaire.pdf
- The musicianship questionnaire we used in this study (reference: Whiteford, K. L., Baltzell, L. S., Chiu, M., Cooper, J. K., Faucher, S., Goh, P. Y., ... & Oxenham, A. J. (2025). Large-scale multi-site study shows no association between musical training and early auditory neural sound encoding. Nature Communications, 16(1), 7152.)
