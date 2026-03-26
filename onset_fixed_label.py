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
training_subjects = ['P01', 'P04', 'P06', 'P07', 'P09', 'P11', 'P12', 'P14']

#force windows to 128 samples even though they only contain 25 actual samples since eegnet expects/is validated with it
fs = 128
original_window_samples = int(round(200 / 1000.0 * fs))
padded_window_samples = 128 
step_samples = original_window_samples // 2  #slide by half of real window
max_timesteps = 160 #enough timesteps to allow longest full song duration to fit
min_timesteps = 10 #was running up against 0 length sequences
output_dim = 1
pretrain_weights_path = "eegnet_listening_pretrain.keras"

#these are a little arbitrary, although smaller batch sizes helped with computation
batch_size_pretrain = 64
epochs_pretrain = 80
lstm_units = 64
batch_size_finetune = 2
epochs_finetune = 30
gaussian_sigma = 0.05

np.random.seed(42)
tf.random.set_seed(42)


class EEGWindowGenerator(Sequence):

    def __init__(self, X, y, indices, batch_size, shuffle=True):
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


def imagination_dataset_generator(data, labels, weights):
    for i, (d, label) in enumerate(zip(data, labels)):
        weight = weights[i]
        yield d, label, weight


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


def create_windows(trial_data):
    n = trial_data.shape[1]
    starts = list(range(0, n - original_window_samples + 1, step_samples))
    if len(starts) == 0:
        return np.zeros((0, trial_data.shape[0], original_window_samples), dtype=np.float32)
    windows = []
    for s in starts:
        windows.append(trial_data[:, s:s+original_window_samples].astype(np.float32))
    return np.stack(windows)


def create_labels(trial_len_samples, stim_onsets_seconds):

    n_windows = 1 + (trial_len_samples - original_window_samples) // step_samples
    onset_samples = (np.array(stim_onsets_seconds) * fs).astype(int)
    starts = np.arange(0, n_windows * step_samples, step_samples)
    centers = starts + (original_window_samples // 2)
    centers_s = centers.astype(float) / fs
    labels = np.zeros((n_windows,), dtype=np.float32)

    if len(onset_samples) == 0:
        return labels.reshape(-1, 1)

    onset_times_s = np.array(stim_onsets_seconds)
    for i, center in enumerate(centers_s):
        diffs = center - onset_times_s
        gaussian = np.exp(-0.5 * (diffs / gaussian_sigma)**2)
        if gaussian.size > 0:
            labels[i] = gaussian.max()
        else:
            labels[i] = 0.0

    return labels.reshape(-1, 1)


def pretrain_dataset(subjects, onset_dict):

    temp_windows = []
    temp_labels = []
    subjects_list = []

    for subj in subjects:
        data, trial_labels = load_subject_npz(subj)
        for trial_idx in range(trial_labels.shape[0]):
            event_id = int(trial_labels[trial_idx])
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 1:  #only listening trials for pretraining
                continue

            trial_music_only = data[trial_idx]  #(chans, samples)
            n_samples = trial_music_only.shape[1]

            raw_windows = create_windows(trial_music_only)
            if subj in ['P01','P04','P06','P07']:
                version = 'v1'
            else:
                version = 'v2'

            onsets = onset_dict[f"{stim_id}_{version}"]
            window_labels = create_labels(n_samples, onsets)

            #trim if labels and data sequences somehow don't match
            if window_labels.shape[0] != raw_windows.shape[0]:
                nmin = min(window_labels.shape[0], raw_windows.shape[0])
                raw_windows = raw_windows[:nmin]
                window_labels = window_labels[:nmin]

            #pad each raw window to total required window samples for eegnet
            pad_width = padded_window_samples - original_window_samples
            for i, w in enumerate(raw_windows):
                w_padded = np.pad(w, ((0,0),(0,pad_width)), mode='constant', constant_values=0.0)
                temp_windows.append(w_padded)  #(chans, window_samples)
                temp_labels.append(window_labels[i])
                subjects_list.append(subj) 

    if len(temp_windows) == 0:
        return np.zeros((0 , 0, padded_window_samples, 1)), np.zeros((0,1), dtype=np.float32), []

    X_raw = np.stack(temp_windows).astype(np.float32)
    X_raw = X_raw[..., np.newaxis]  #(N, chans, window_samples, 1)

    #normalize
    mean = X_raw.mean(axis=(0,1,2), keepdims=True)
    std  = X_raw.std(axis=(0,1,2), keepdims=True)
    np.savez("global_norm.npz", mean=mean, std=std)
    Xs = (X_raw - mean) / (std + 1e-12)
    labels = np.stack(temp_labels).astype(np.float32)

    return Xs, labels, subjects_list


def training_dataset(subjects, onset_dict, mean, std):

    sequences = []
    temp_labels = []
    masks = []
    metadata = []

    for subj in subjects:
        data, trial_labels = load_subject_npz(subj)
        for trial_idx in range(trial_labels.shape[0]):
            event_id = int(trial_labels[trial_idx])
            stim_id = event_id // 10
            condition_num = event_id % 10
            if condition_num != 2:  #only want imagination sequences for regular training
                continue

            trial_music_only = data[trial_idx]
            n_samples = trial_music_only.shape[1]

            raw_windows = create_windows(trial_music_only)
            pad_width = padded_window_samples - original_window_samples
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
            window_labels = create_labels(n_samples, onsets)

            if window_labels.shape[0] != windows.shape[0]:
                min_size = min(window_labels.shape[0], windows.shape[0])
                windows = windows[:min_size]
                window_labels = window_labels[:min_size]

            if windows.shape[0] < min_timesteps:
                continue

            sequences.append(windows.astype(np.float32))
            temp_labels.append(window_labels.astype(np.float32))
            masks.append(np.ones(windows.shape[0], dtype=np.float32))
            metadata.append((subj, trial_idx, int(stim_id)))
    
    return sequences, temp_labels, masks, metadata



def pretrain_eegnet(X_listen, y_listen, subjects_list, save_path):

    chans = X_listen.shape[1]
    samples = X_listen.shape[2]
    train_subjects, val_subjects = train_test_split(subjects_list, test_size=0.1, random_state=42)
    train_inds = [i for i, s in enumerate(subjects_list) if s in train_subjects]
    val_inds = [i for i, s in enumerate(subjects_list) if s in val_subjects]
    train_gen = EEGWindowGenerator(X_listen, y_listen, train_inds, batch_size_pretrain, shuffle=True)
    val_gen = EEGWindowGenerator(X_listen, y_listen, val_inds, batch_size_pretrain, shuffle=False)

    #build model, but only using cnn plus dense layer, not cnn to two lstms to dense layer unlike regular training. should I be using lstms here too?
    base = EEGNet(nb_classes=output_dim, Chans=chans, Samples=samples)
    flatten_layer = base.get_layer('flatten').output 
    out = Dense(output_dim, activation='linear', name='regression_output')(flatten_layer)
    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse', metrics=['mse'])
    checkpoint = ModelCheckpoint(save_path, save_best_only=True, monitor='val_loss', mode='min')
    early = EarlyStopping(monitor='val_loss', patience=10, mode='min', restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=epochs_pretrain, callbacks=[checkpoint, early])


def build_cnn_rnn(channels, samples):

    base = EEGNet(nb_classes=output_dim, Chans=channels, Samples=samples)
    base.load_weights(pretrain_weights_path)
    flatten_layer = base.get_layer('flatten')   #cut off before classification part
    eegnet_feat_model = Model(inputs=base.input, outputs=flatten_layer.output)

    seq_input = Input(shape=(None, channels, samples, 1), name='seq_input') #(batch, timesteps, chans, samples, 1)

    # def apply_eegnet(x):
    #     batch_size = tf.shape(x)[0]
    #     num_timesteps = tf.shape(x)[1]
    #     reshaped_x = tf.reshape(x, (batch_size * num_timesteps, eegnet_feat_model.input_shape[1], eegnet_feat_model.input_shape[2], 1))
    #     feats = eegnet_feat_model(reshaped_x)   #(batch_size*num_timesteps, feat_dim)
    #     feat_dim = eegnet_feat_model.output_shape[1]
    #     feats_seq = tf.reshape(feats, (batch_size, num_timesteps, feat_dim))
    #     return feats_seq

    # feats = Lambda(apply_eegnet, name='eegnet_time_dist', mask=lambda inputs, mask: mask)(seq_input)  #(batch, timesteps, feat_dim)
    feats = TimeDistributed(eegnet_feat_model, name='eegnet_time_dist')(seq_input)

    masked_feats = Masking(mask_value=0.0)(feats) #mask so lstm doesn't learn padded zeroes as features
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(masked_feats)
    x = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    out = TimeDistributed(Dense(output_dim, activation='linear'))(x)
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer=Adam(1e-3), loss='mse')
    return model


def yield_data(sequences, labels, sample_weights, batch_size, shuffle=True):

    '''
    creating the datasets altogether used too much ram, yielding this way helped
    '''

    output_signature = (tf.TensorSpec(shape=(None, sequences[0].shape[1], sequences[0].shape[2], 1), dtype=tf.float32), tf.TensorSpec(shape=(None, 1), dtype=tf.float32), tf.TensorSpec(shape=(None,), dtype=tf.float32))
    dataset = tf.data.Dataset.from_generator(lambda: imagination_dataset_generator(sequences, labels, sample_weights), output_signature=output_signature)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=min(16, len(sequences)))
    dataset = dataset.padded_batch(
        batch_size,
        padded_shapes=([None, sequences[0].shape[1], sequences[0].shape[2], 1], [None, 1], [None] ),
        padding_values=(0.0, 0.0, 0.0)
    )
    return dataset.prefetch(tf.data.AUTOTUNE)



onset_dict = {}
for stim_id in [1, 2, 3, 4, 11, 12, 13, 14, 21, 22, 23, 24]:
    #need v1 and v2 since MIDI changed for later participants (slight tempo shifts, see openmiir paper)
    found_v1 = os.path.join('/content', f"stim_{stim_id}_v1.mid") 
    found_v2 = os.path.join('/content', f"stim_{stim_id}_v2.mid")
    onset_dict[str(stim_id) + "_v1"] = extract_midi_onsets(found_v1)
    onset_dict[str(stim_id) + "_v2"] = extract_midi_onsets(found_v2)

X_listen, y_listen, subjects_list = pretrain_dataset(training_subjects, onset_dict)
pretrain_eegnet(X_listen, y_listen, subjects_list, save_path=pretrain_weights_path)

norm = np.load("global_norm.npz")
mean_pre = norm["mean"]  #(1, chans, samples, 1)
std_pre  = norm["std"]

train_sequences, train_labels, train_masks, train_metadata = training_dataset(training_subjects, onset_dict, mean_pre, std_pre)
test_sequences, test_labels, test_masks, test_metadata = training_dataset([holdout_subj], onset_dict, mean_pre, std_pre)
train_subjs, val_subjs = train_test_split(training_subjects, test_size=0.2, random_state=42)

train_sequences_split, train_labels_split, train_masks_split = [], [], []
val_sequences_split, val_labels_split, val_masks_split = [], [], []

for seq, label, mask, metadata in zip(train_sequences, train_labels, train_masks, train_metadata):
    subj = metadata[0]
    if subj in train_subjs:
        train_sequences_split.append(seq)
        train_labels_split.append(label)
        train_masks_split.append(mask)
    elif subj in val_subjs:
        val_sequences_split.append(seq)
        val_labels_split.append(label)
        val_masks_split.append(mask)

chans = train_sequences_split[0].shape[1]
samples = train_sequences_split[0].shape[2]
model = build_cnn_rnn(chans, samples)

train_ds = yield_data(train_sequences_split, train_labels_split, train_masks_split, batch_size_finetune, shuffle=True)
val_ds = yield_data(val_sequences_split, val_labels_split, val_masks_split, batch_size_finetune, shuffle=False)
test_ds = yield_data(test_sequences, test_labels, test_masks, 1, shuffle=False)

ckpt = ModelCheckpoint(f"finetune_onset.keras", save_best_only=True, monitor='val_loss')
early = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True) #should we get rid of earlystopping? I think it's just a time saver if I understand right, not helping performance necessarily. I think not the biggest priority though
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs_finetune, callbacks=[ckpt, early])

plt.figure()
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title(f"Onset P13 Training vs Validation Loss")
plt.legend()
plt.savefig(f'/content/drive/MyDrive/training_vs_validation_loss_p13_onset.png')
plt.close()

eval_loss = model.evaluate(test_ds)
model.load_weights("finetune_onset.keras") #does it matter if weights load before/after evaluating model?
preds = model.predict(test_ds)

print(f"P13 onset test loss (MSE): {eval_loss}")
