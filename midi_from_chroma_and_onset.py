import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from scipy.ndimage import gaussian_filter1d
import pretty_midi
import sys
sys.path.append('./eeg_models')
from eeg_models.EEGModels import EEGNet
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from keras.saving import register_keras_serializable



fs = 128
onset_norm = np.load("global_norm_onset.npz")
mean_onset = onset_norm["mean"]
std_onset = onset_norm["std"]
chroma_norm = np.load("global_norm_chroma.npz")
mean_chroma = chroma_norm["mean"]
std_chroma = chroma_norm["std"]

onset_params = {
    "original_window_samples": int(round(200/1000 * fs)), 
    "padded_window_samples": 128,
    "step_samples": int(round(200/1000 * fs))//2, 
    "max_timesteps": 160,
    "gaussian_sigma": 0.05,
}

chroma_params = {
    "original_window_samples": int(round(400/1000 * fs)),
    "padded_window_samples": 128,
    "step_samples": int(round(400/1000 * fs))//2,
    "max_timesteps": 100,
}

song_lengths_v2 = {
    1: 13.301,
    2: 7.7,
    3: 9.7,
    4: 11.6,
    11: 13.5,
    12: 7.7,
    13: 9.0,
    14: 12.2,
    21: 8.275,
    22: 16.000,
    23: 9.2,
    24: 6.956,
}


def load_subject_npz(subj):
    p = os.path.join('/content', f"{subj}-raw_data_v2.npz")
    npz = np.load(p)
    return npz["data"], npz["labels"]


def prepare_windows(trial, params, mean, std):

    n_samples = trial.shape[1] #trial shape is (channels, time)
    original_window_samples = params["original_window_samples"]

    #window trial data
    starts = list(range(0, n_samples - original_window_samples + 1, params["step_samples"]))
    windows = []
    for s in starts:
        windows.append(trial[:, s:s+original_window_samples].astype(np.float32)) #takes all channels (first trial dimension) but only enough samples past start point to fill window size
    raw_windows = np.stack(windows)
    
    #pad windows to length required by eegnet
    pad_width = params["padded_window_samples"] - params["original_window_samples"]    
    win_list = []
    for w in raw_windows:
        w_padded = np.pad(w, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
        w_padded = w_padded[np.newaxis,...,np.newaxis] #now shape is (batch_size, channels, time steps, features)
        w_norm = (w_padded - mean)/(std + 1e-12)
        win_list.append(w_norm[0]) #had to add batch_size for mean/std dimensions, but can remove now bc we don't need it
    
    windows = np.stack(win_list).astype(np.float32) #shape is (num_windows, channels, padded_window_samples, 1)

    return windows[:params["max_timesteps"]], n_samples


'''
below model loading is using eegnet with pretrained weights, but training code allowed eegnet to be updated
during regular training too. maybe pretrained eegnet is overriding finetune weights? will try that instead
and see if results improve

also for build_eegnet_feature_extractor, chroma used softmax, but I think onset used linear activation? look into this
'''

def build_eegnet_feature_extractor(chans, samples, nb_classes, pretrained_weights):

    base = EEGNet(nb_classes=nb_classes, Chans=chans, Samples=samples)
    flatten_layer = base.get_layer('flatten').output #we don't necessarily want eegnet classifier output, so take layer before that which is flatten layer
    out = tf.keras.layers.Dense(nb_classes, activation='softmax')(flatten_layer)
    model = Model(inputs=base.input, outputs=out)
    model.load_weights(pretrained_weights)
    feat_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    return feat_model


eegnet_model_onset = build_eegnet_feature_extractor(chans=64, samples=128, nb_classes=1, pretrained_weights='/content/pretrain_onset.keras')
@register_keras_serializable()
def apply_eegnet_onset(x):
    
    batch_size = tf.shape(x)[0]
    num_timesteps = tf.shape(x)[1]
    x_resh = tf.reshape(x, (batch_size * num_timesteps, eegnet_model_onset.input_shape[1], eegnet_model_onset.input_shape[2], 1))
    
    feats = eegnet_model_onset(x_resh)
    feat_dim = eegnet_model_onset.output_shape[1]
    feats_seq = tf.reshape(feats, (batch_size, num_timesteps, feat_dim))
    
    return feats_seq


eegnet_model_chroma = build_eegnet_feature_extractor(chans=64, samples=128, nb_classes=12, pretrained_weights='/content/pretrain_chroma.keras')
@register_keras_serializable()
def apply_eegnet_chroma(x):
    
    batch_size = tf.shape(x)[0]
    num_timesteps = tf.shape(x)[1]
    reshaped_x = tf.reshape(x, (batch_size * num_timesteps, eegnet_model_chroma.input_shape[1], eegnet_model_chroma.input_shape[2], 1))
    
    feats = eegnet_model_chroma(reshaped_x)
    feat_dim = eegnet_model_chroma.output_shape[1]
    feats_seq = tf.reshape(feats, (batch_size, num_timesteps, feat_dim))
    
    return feats_seq


onset_model = load_model("finetune_onset.keras", custom_objects={'apply_eegnet': apply_eegnet_onset})
chroma_model = load_model("finetune_chroma.keras", custom_objects={'apply_eegnet': apply_eegnet_chroma})
data, labels = load_subject_npz('P13')



def create_predictions(labels):

    with tf.device("/GPU:0"):
        results = {}

        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])
            stim_id = event_id // 10
            condition_num = event_id % 10

            if condition_num != 2:  #only imagination trials with cue clicks immediately before
                continue
            if stim_id in results: #we've already seen one iteration of this trial
                continue

            trial = data[trial_idx]

            onset_windows, onset_len = prepare_windows(trial, onset_params, mean_onset, std_onset)
            onset_input = onset_windows[np.newaxis, ...]  #add batch dim since needed for prediction, giving (1, timesteps, chans, samples, 1)
            onset_pred = onset_model.predict(onset_input, batch_size=32) #gives dimensions (1, timesteps, 1)
            onset_pred = onset_pred.squeeze(0)  #remove batch dim after prediction, giving (timesteps, 1)

            chroma_windows, chroma_len = prepare_windows(trial, chroma_params, mean_chroma, std_chroma)
            chroma_input = chroma_windows[np.newaxis, ...] 
            chroma_pred = chroma_model.predict(chroma_input, batch_size=32)
            chroma_pred = chroma_pred.squeeze(0)

            plt.figure()
            plt.plot(onset_pred.squeeze(), label="onset_pred")
            plt.title(f"Stimulus {stim_id} onset prediction")
            plt.xlabel("Timesteps")
            plt.ylabel("Predicted onset probability")
            plt.savefig(f'/content/drive/MyDrive/stim{stim_id}_onset_prediction_fixed_labels.png')
            plt.close()

            plt.figure()
            plt.imshow(chroma_pred.T, aspect="auto", origin="lower")
            plt.title(f"Stimulus {stim_id} chroma prediction")
            plt.xlabel("Timesteps")
            plt.ylabel("Pitch class")
            plt.colorbar()
            plt.savefig(f'/content/drive/MyDrive/stim{stim_id}_chroma_prediction_fixed_labels.png')
            plt.close()

            results[stim_id] = {"onset_pred": onset_pred, "chroma_pred": chroma_pred, "trimmed_len": onset_len}

        return results


def refine_peak_time(onset_curve, idx, centers):

    '''
    interpolates peaks to more accurately estimate onset times
    '''

    if idx <= 0 or idx >= len(onset_curve) - 1:
        return centers[idx]

    #used formula from this link to interpolate peaks: https://www.dsprelated.com/freebooks/sasp/Quadratic_Interpolation_Spectral_Peaks.html 
    y1 = onset_curve[idx-1]
    y2 = onset_curve[idx]
    y3 = onset_curve[idx+1]
    denom = (y1 - 2*y2 + y3)
    offset = 0.5 * (y1 - y3) / (denom + 1e-12)

    step = centers[1] - centers[0]
    refined_time = centers[idx] + offset * step

    return refined_time


def pred_to_midi(preds):
    
    '''
    generated MIDI based on note onset and, if there's a detected onset, predicted pitch
    '''

    onset_pred = preds["onset_pred"]
    chroma_pred = preds["chroma_pred"]
    trimmed_len = preds["trimmed_len"]
    chroma_lookahead = 0.15 #roughly eighth note length?

    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0)

    trial_duration = trimmed_len / fs

    #calculates center of each window in terms of seconds
    centers_onset = (np.arange(len(onset_pred)) * onset_params["step_samples"] + onset_params["original_window_samples"] // 2) / fs
    centers_chroma = (np.arange(len(chroma_pred)) * chroma_params["step_samples"] + chroma_params["original_window_samples"] // 2) / fs

    #commented out line has never been used but would smooth onset predictions, not sure if you actually want this bc idk if you lose actual onset peaks (especially bc onset is already underestimating)
    # onset_smoothed = gaussian_filter1d(onset_pred.squeeze(), sigma=1)
    onset_smoothed = onset_pred.squeeze()

    #not sure if defining height and distance params in find_peaks could help, although model is under-predicting rn
    #also maybe the props could be helpful somehow?
    onset_indices, props = find_peaks(onset_smoothed)

    for i, onset_index in enumerate(onset_indices):
        start_time = refine_peak_time(onset_smoothed, onset_index, centers_onset)
        if start_time >= trial_duration: #trim predictions to actual trial length (currently predicting too long)
            continue

        #look at chroma prediction slightly after onset time
        chroma_target = start_time + chroma_lookahead
        chroma_idx = np.argmin(np.abs(centers_chroma - chroma_target))
        chroma_vec = chroma_pred[chroma_idx]
        pitch_class = int(np.argmax(chroma_vec))

        #set duration until next note starts or piece ends
        if i < len(onset_indices)-1:
            end_time = refine_peak_time(onset_smoothed, onset_indices[i+1], centers_onset)
        else:
            end_time = trial_duration

        note = pretty_midi.Note(pitch = 60 + pitch_class, velocity = 60, start = start_time, end = end_time)
        piano.notes.append(note)

    pm.instruments.append(piano)
    return pm


def only_chroma_to_midi(preds):

    '''
    this function generates MIDI by outputting the chroma prediction at every time step, ignoring onset
    '''

    chroma_pred = preds["chroma_pred"]
    trimmed_len_samples = preds["trimmed_len"]

    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(0)

    trial_duration = trimmed_len_samples / fs

    centers_chroma = (np.arange(len(chroma_pred)) * chroma_params["step_samples"] + chroma_params["original_window_samples"] // 2) / fs

    for i in range(len(chroma_pred)):

        start_time = centers_chroma[i]
        if start_time >= trial_duration: 
            continue

        #end note at next timestep
        if i < len(chroma_pred) - 1:
            end_time = centers_chroma[i + 1]
        else:
            end_time = trial_duration

        chroma_vec = chroma_pred[i]
        pitch_class = int(np.argmax(chroma_vec))
        note = pretty_midi.Note(pitch= 60 + pitch_class, velocity = 60, start = start_time, end = end_time)
        piano.notes.append(note)

    pm.instruments.append(piano)
    return pm


def trim_midi_to_duration(pm, max_duration):

    '''
    trims MIDI to known piece length, since model is predicting longer segments
    '''

    for inst in pm.instruments:
        new_notes = []
        for note in inst.notes:
            if note.start >= max_duration: #drop notes that start after cutoff
                continue
            note.end = min(note.end, max_duration)
            new_notes.append(note)
        inst.notes = new_notes


results = create_predictions(labels)
output_dir = "/content/reconstructed_midi"
os.makedirs(output_dir, exist_ok=True)

for stim_id, preds in results.items():
    pm = pred_to_midi(preds)
    # pm = only_chroma_to_midi(preds)
    song_duration = song_lengths_v2[stim_id]
    trim_midi_to_duration(pm, song_duration) #in theory, I've done the trimming in pred_to_midi function. but just in case
    midi_path = os.path.join(output_dir, f"P13_stim{stim_id}.mid")
    pm.write(midi_path)
