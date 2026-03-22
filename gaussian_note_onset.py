import os
import pretty_midi
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.utils import Sequence
from tensorflow.keras.layers import Input, Dense, Bidirectional, LSTM, TimeDistributed, Lambda, Masking
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import sys
sys.path.append('./eeg_models')
from eeg_models.EEGModels import EEGNet

   
subjects = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P13', 'P14']
holdout_subj = 'P13'
training_subjects = [s for s in subjects if s != holdout_subj]

#force windows to 128 samples even though they only contain 25 actual samples since eegnet expects it
fs = 128
original_window_samples = int(round(200 / 1000.0 * fs))
window_samples = 128 
step_samples = original_window_samples // 2  #slide by half of real window
max_timesteps = 160
min_timesteps = 10

pretrain_weights_path = "eegnet_listening_pretrain.keras"
batch_size_pretrain = 64
epochs_pretrain = 80

lstm_units = 64
batch_size_finetune = 2
epochs_finetune = 30
output_dim = 1

gaussian_sigma = 0.05
stim_ids = [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]

np.random.seed(42)
tf.random.set_seed(42)


class EEGWindowGenerator(Sequence):

    def __init__(self, X, y, indices, batch_size=32, shuffle=True):
        super().__init__()
        self.X = X
        self.y = y
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_inds = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        X_batch = self.X[batch_inds]
        y_batch = self.y[batch_inds]
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)


def imagination_dataset_generator(sequences, targets, sample_weights):
    for i, (seq, tgt) in enumerate(zip(sequences, targets)):
        sw = sample_weights[i]
        yield seq, tgt, sw


def load_subject_npz(subj):
    p = os.path.join('/content', f"{subj}-raw_data_v2.npz")
    npz = np.load(p)
    return npz["data"], npz["labels"]


def extract_midi_onsets(midi_path):
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets = []
    for inst in pm.instruments:
        for n in inst.notes:
            onsets.append(n.start)
    onsets = np.unique(np.array(onsets))
    return np.sort(onsets)


def build_midi_onset_dict():
    midi_dir = '/content'
    onset_dict = {}
    for stim_id in stim_ids:
        #need v1 and v2 since MIDI changed for different participants
        found_v1 = os.path.join(midi_dir, f"stim_{stim_id}_v1.mid") 
        found_v2 = os.path.join(midi_dir, f"stim_{stim_id}_v2.mid")
        onset_times_v1 = extract_midi_onsets(found_v1)
        onset_times_v2 = extract_midi_onsets(found_v2)
        onset_dict[str(stim_id) + "_v1"] = onset_times_v1
        onset_dict[str(stim_id) + "_v2"] = onset_times_v2
    return onset_dict


def sliding_windows_from_trial(trial_data):
    n = trial_data.shape[1]
    starts = list(range(0, n - original_window_samples + 1, step_samples))
    if len(starts) == 0:
        return np.zeros((0, trial_data.shape[0], original_window_samples), dtype=np.float32)
    windows = []
    for s in starts:
        windows.append(trial_data[:, s:s+original_window_samples].astype(np.float32))
    return np.stack(windows)


def build_regression_targets_for_trial(trial_len_samples, stim_onsets_seconds):

    n_windows = 1 + (trial_len_samples - original_window_samples) // step_samples
    onset_samples = (np.array(stim_onsets_seconds) * fs).astype(int)
    starts = np.arange(0, n_windows * step_samples, step_samples)
    centers = starts + (original_window_samples // 2)
    centers_s = centers.astype(float) / fs
    targets = np.zeros((n_windows,), dtype=np.float32)

    if len(onset_samples) == 0:
        return targets.reshape(-1, 1)

    onset_times_s = np.array(stim_onsets_seconds)
    for i, ct in enumerate(centers_s):
        diffs = ct - onset_times_s
        g = np.exp(-0.5 * (diffs / gaussian_sigma)**2)
        if g.size > 0:
            targets[i] = g.max()
        else:
            targets[i] = 0.0

    return targets.reshape(-1, 1)


def build_listening_pretrain_dataset_onset(subjects, onset_dict):

    temp_windows = []
    temp_targets = []
    subjects_list = []

    for subj in subjects:
        data, labels = load_subject_npz(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 1:  #only listening trials for pretraining
                continue

            trial_music_only = data[trial_idx]  #(chans, samples)
            n_samples = trial_music_only.shape[1]

            raw_windows = sliding_windows_from_trial(trial_music_only)
            if raw_windows.shape[0] == 0:
                continue

            if subj in ['P01','P04','P06','P07']:
                version = 'v1'
            else:
                version = 'v2'

            onsets = onset_dict[f"{stim_id}_{version}"]

            y = build_regression_targets_for_trial(trial_len_samples=n_samples, stim_onsets_seconds=onsets)

            #trim in case of mismatch
            if y.shape[0] != raw_windows.shape[0]:
                nmin = min(y.shape[0], raw_windows.shape[0])
                raw_windows = raw_windows[:nmin]
                y = y[:nmin]

            #pad each raw window to total required window samples
            pad_width = window_samples - original_window_samples
            for i, w in enumerate(raw_windows):
                w_padded = np.pad(w, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
                temp_windows.append(w_padded)  #(chans, window_samples)
                temp_targets.append(y[i])
                subjects_list.append(subj) 

    if len(temp_windows) == 0:
        return np.zeros((0,0,window_samples,1)), np.zeros((0,1), dtype=np.float32), []

    X_raw = np.stack(temp_windows).astype(np.float32)
    X_raw = X_raw[..., np.newaxis]  #(N, chans, window_samples, 1)

    #normalize
    mean = X_raw.mean(axis=(0,1,2), keepdims=True)
    std  = X_raw.std(axis=(0,1,2), keepdims=True)
    np.savez("global_norm.npz", mean=mean, std=std)
    Xs = (X_raw - mean) / (std + 1e-12)
    y = np.stack(temp_targets).astype(np.float32)

    return Xs, y, subjects_list


def build_imagination_sequences(subjects, onset_dict, mean, std):

    sequences = []
    targets = []
    masks = []
    metadata = []

    for subj in subjects:
        data, labels = load_subject_npz(subj)
        for trial_idx in range(labels.shape[0]):
            event_id = int(labels[trial_idx])
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 2:  #only want imagination sequences for regular training
                continue

            trial_music_only = data[trial_idx]
            n_samples = trial_music_only.shape[1]

            raw_windows = sliding_windows_from_trial(trial_music_only)
            if raw_windows.shape[0] == 0:
                continue

            pad_width = window_samples - original_window_samples
            win_list = []
            for w in raw_windows:
                w_padded = np.pad(w, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)  #(chans, window_samples)
                w_padded = w_padded[np.newaxis, ... , np.newaxis]  #change shape to (1, chans, window_samples, 1) to match mean/std requirements
                w_norm = (w_padded - mean) / (std + 1e-12)
                win_list.append(w_norm[0])  #(chans, window_samples, 1)
            windows = np.stack(win_list, axis=0).astype(np.float32)  #(timesteps, chans, window_samples, 1)

            #capping sequences length has helped a lot with computation time/requirements
            if windows.shape[0] > max_timesteps:
                windows = windows[:max_timesteps]

            if subj in ['P01','P04','P06','P07']:
                version = 'v1'
            else:
                version = 'v2'
            onsets = onset_dict[f"{stim_id}_{version}"]
            y = build_regression_targets_for_trial(trial_len_samples=n_samples, stim_onsets_seconds=onsets)

            if y.shape[0] != windows.shape[0]:
                nmin = min(y.shape[0], windows.shape[0])
                windows = windows[:nmin]
                y = y[:nmin]

            if windows.shape[0] < min_timesteps:
                continue

            sequences.append(windows.astype(np.float32))
            targets.append(y.astype(np.float32))
            masks.append(np.ones(windows.shape[0], dtype=np.float32))
            metadata.append((subj, trial_idx, int(stim_id)))
    
    #make sure there aren't 0-length sequences just in case
    filtered_sequences = []
    filtered_targets = []
    filtered_masks = []
    filtered_metadata = []
    
    for s, t, m, md in zip(sequences, targets, masks, metadata):
        if s.shape[0] > 0:
            filtered_sequences.append(s)
            filtered_targets.append(t)
            filtered_masks.append(m)
            filtered_metadata.append(md)

    return filtered_sequences, filtered_targets, filtered_masks, filtered_metadata



def pretrain_eegnet_onset(X_listen, y_listen, subjects_list, save_path):

    chans = X_listen.shape[1]
    samples = X_listen.shape[2]

    unique_subjects = np.unique(subjects_list)
    train_subjects, val_subjects = train_test_split(unique_subjects, test_size=0.1, random_state=42)

    train_inds = [i for i, s in enumerate(subjects_list) if s in train_subjects]
    val_inds = [i for i, s in enumerate(subjects_list) if s in val_subjects]

    train_gen = EEGWindowGenerator(X_listen, y_listen, train_inds, batch_size=batch_size_pretrain, shuffle=True)
    val_gen = EEGWindowGenerator(X_listen, y_listen, val_inds, batch_size=batch_size_pretrain, shuffle=False)

    base = EEGNet(nb_classes=output_dim, Chans=chans, Samples=samples)
    flatten_layer = base.get_layer('flatten').output 
    out = Dense(output_dim, activation='linear', name='regression_output')(flatten_layer)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mse'])

    checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min')
    early = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)

    model.fit(train_gen, validation_data=val_gen, epochs=epochs_pretrain, callbacks=[checkpoint, early])

    return save_path


def build_eegnet_feature_extractor(chans, samples):
    base = EEGNet(nb_classes=output_dim, Chans=chans, Samples=samples)
    base.load_weights(pretrain_weights_path)
    flatten_layer = base.layers[-3]   #cut off before classification part
    feat_model = Model(inputs=base.input, outputs=flatten_layer.output)
    return feat_model


def build_cnn_rnn_model(eegnet_feat_model):

    seq_input = Input(shape=(None, eegnet_feat_model.input_shape[1], eegnet_feat_model.input_shape[2], 1), name='seq_input') #(batch, timesteps, chans, samples, 1)

    def apply_eegnet(x):
        batch_size = tf.shape(x)[0]
        num_timesteps = tf.shape(x)[1]
        reshaped_x = tf.reshape(x, (batch_size * num_timesteps, eegnet_feat_model.input_shape[1], eegnet_feat_model.input_shape[2], 1))
        feats = eegnet_feat_model(reshaped_x)   #(batch_size*num_timesteps, feat_dim)
        feat_dim = eegnet_feat_model.output_shape[1]
        feats_seq = tf.reshape(feats, (batch_size, num_timesteps, feat_dim))
        return feats_seq

    feats = Lambda(apply_eegnet, name='eegnet_time_dist', mask=lambda inputs, mask: mask)(seq_input)  #(batch, timesteps, feat_dim)
    masked_feats = Masking(mask_value=0.0)(feats) #mask so lstm doesn't learn padded zeroes as features
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(masked_feats)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    out = TimeDistributed(Dense(output_dim, activation='linear'))(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model

def create_tf_dataset(sequences, targets, sample_weights, batch_size, shuffle=True):
    output_signature = (tf.TensorSpec(shape=(None, sequences[0].shape[1], sequences[0].shape[2], 1), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(lambda: imagination_dataset_generator(sequences, targets, sample_weights), output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(16, len(sequences)))
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, sequences[0].shape[1], sequences[0].shape[2], 1], [None, 1], [None] ),
        padding_values=(0.0, 0.0, 0.0)
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    print("Building midi onset dictionary")
    onset_dict = build_midi_onset_dict()

    if not os.path.exists(pretrain_weights_path):
        print("Building listening pretrain dataset")
        X_listen, y_listen, subjects_list = build_listening_pretrain_dataset_onset(training_subjects, onset_dict)
        pretrain_eegnet_onset(X_listen, y_listen, subjects_list, save_path=pretrain_weights_path,)
    else:
        print("Found pretrained weights", pretrain_weights_path)

    norm = np.load("global_norm.npz")
    mean_pre = norm["mean"]  #(1, chans, samples, 1)
    std_pre  = norm["std"]

    print("Building imagination sequences")
    train_sequences, train_targets, train_masks, train_metadata = build_imagination_sequences(training_subjects, onset_dict, mean_pre, std_pre)
    test_sequences, test_targets, test_masks, test_metadata = build_imagination_sequences([holdout_subj], onset_dict, mean_pre, std_pre)

    train_subjs, val_subjs = train_test_split(training_subjects, test_size=0.2, random_state=42)

    train_sequences_split, train_targets_split, train_masks_split = [], [], []
    val_sequences_split, val_targets_split, val_masks_split = [], [], []

    for seq, tgt, mask, metadata in zip(train_sequences, train_targets, train_masks, train_metadata):
        subj = metadata[0]
        if subj in train_subjs:
            train_sequences_split.append(seq)
            train_targets_split.append(tgt)
            train_masks_split.append(mask)
        elif subj in val_subjs:
            val_sequences_split.append(seq)
            val_targets_split.append(tgt)
            val_masks_split.append(mask)

    test_sequences_split = test_sequences
    test_targets_split = test_targets
    test_masks_split = test_masks

    chans = train_sequences_split[0].shape[1]
    samples = train_sequences_split[0].shape[2]

    print("Building eegnet feature extractor")
    eegnet_feat = build_eegnet_feature_extractor(chans, samples)

    print("Building cnn-rnn")
    model = build_cnn_rnn_model(eegnet_feat)
    model.summary()

    print("Creating tf datasets")
    train_ds = create_tf_dataset(train_sequences_split, train_targets_split, train_masks_split, batch_size_finetune, shuffle=True)
    val_ds = create_tf_dataset(val_sequences_split, val_targets_split, val_masks_split, batch_size_finetune, shuffle=False)
    test_ds = create_tf_dataset(test_sequences_split, test_targets_split, test_masks_split, 1, shuffle=False)

    print("Training model")
    ckpt = ModelCheckpoint(f"finetune_onset.keras", save_best_only=True, monitor='val_loss')
    early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune, callbacks=[ckpt, early])

    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"Onset P13 Training vs Validation Loss (Fixed Labels)")
    plt.legend()
    plt.savefig(f'/content/drive/MyDrive/training_vs_validation_loss_p13_onset_fixed_labels.png')
    plt.close()

    eval_loss = model.evaluate(test_ds)
    model.load_weights("finetune_onset.keras")
    preds = model.predict(test_ds)

    print(f"P13 onset test loss (MSE): {eval_loss}")


if __name__ == "__main__":
    main()
