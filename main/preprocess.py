import os
import numpy as np
import pandas as pd
import itertools
import scipy as sp
from scipy import signal
import librosa
from librosa import effects

# GLOBAL VARS
samples_rate = 16 * 10**3
emph_alpha = 0.95

frame_sec_size = 0.025
overlap_sec_size = 0.01
nfft = int(samples_rate * frame_sec_size)
win_len = nfft
hop_len = int(samples_rate * overlap_sec_size)
num_segments = 2
num_features = nfft // 2 + 1
hamming = sp.signal.windows.hamming(win_len)

top_decibells = 20
quant = 0.9


def get_dataset_frame(audio_direct, ext, num_speakers):
    speak_with_dirs = [(e[0], e[2]) for e in os.walk(audio_direct)]
    # Remove non-parents dirs
    speak_with_dirs = [e for e in speak_with_dirs
                       if len(e[1]) > 0]
    # Get closest path = it's path to parent dir
    parents = list(map(lambda e: e[0], speak_with_dirs))

    parents = list(map(lambda s: s.replace('\\', '/'), parents))

    chapters = list(map(lambda s: s.split('/')[-1], parents))
    # Last dir in parent path = speaker's target
    targets = list(map(lambda s: s.split('/')[-2], parents))

    # Paths without targets
    paths = list(map(lambda s: '/'.join(s.split('/')[:-1]), parents))
    # Associate with speaker files
    wavs = list(map(lambda e: e[1], speak_with_dirs))
    wavs = [list(filter(lambda s: s.endswith(ext), e))
            for e in wavs]
    # Connect files with relevant target and filepaths
    wavs = [[(path, file, target, chapter) for file in files]
            for target, files, path, chapter in zip(targets, wavs, paths, chapters)]
    wavs = list(itertools.chain(*wavs))
    df_all = pd.DataFrame(wavs, columns=[
        'Parent', 'File', 'Target', 'Chapter'
    ])
    fulls = list(map(
        lambda e: '/'.join(e),
        zip(df_all['Parent'], df_all['Chapter'], df_all['File'])
    ))
    df_all['Full_path'] = fulls
    target_vc = df_all['Target'].value_counts()
    # Select n most popular speakers by id
    top_speaks = target_vc[:num_speakers]
    top_ids = top_speaks.index
    df = df_all[df_all['Target'].isin(top_ids)]
    df.index = range(len(df))
    return df_all, df, top_ids


def get_triplets(audio_direct, ext, num_speakers, triplet_len):
    df_all, df, labels = get_dataset_frame(audio_direct, ext, num_speakers)
    # a - Anchors and positives collection
    a = []
    for label in labels:
        b = df[df['Target'] == label]
        a.append(b)

    df1 = pd.concat(a)
    # Collections of anchors and positives examples paths
    anchors = []
    positives = []

    for label in labels:
        sub_df = df1.loc[df1['Target'] == label,
                         ['Full_path', 'Target']]

        anchor = sub_df.iloc[:triplet_len]
        positive = sub_df.iloc[triplet_len: 2 * triplet_len]

        anchors.append(anchor['Full_path'].to_numpy())
        positives.append(positive['Full_path'].to_numpy())

    anchors = np.array(list(itertools.chain(*anchors)))
    positives = np.array(list(itertools.chain(*positives)))

    # Dataset length
    samples_count = len(anchors)

    # Select random samples of rest dataset
    # to build impostor set
    neg_df = df_all[~df_all.index.isin(df.index)]
    imposts = neg_df.sample(samples_count, random_state=5)
    negatives = imposts['Full_path'].to_numpy()

    anchors = anchors.reshape((samples_count, 1))
    positives = positives.reshape((samples_count, 1))
    negatives = negatives.reshape((samples_count, 1))

    dataset = np.concatenate([anchors, positives, negatives], axis=1)
    return dataset.ravel(), samples_count, labels


def load_emphas(path, alpha, rate):
    audio, _ = librosa.load(path, sr=rate)
    audio = effects.preemphasis(audio, coef=alpha)
    return audio


def clear_from_silence(wave):
    sounded_ints = effects.split(
        wave, top_db=top_decibells,
        frame_length=win_len, hop_length=hop_len
    )
    sounded_wave = [wave[inter[0]:inter[1]]
                    for inter in sounded_ints]
    return np.concatenate(sounded_wave)


def get_threshold_duration(waves):
    lens = [len(wave) for wave in waves]
    return int(np.quantile(lens, q=quant))


def truncate_or_pad(audio_direct, ext, num_speakers,
                    triplet_len, alpha, rate):
    paths, samples_count, labels = get_triplets(
        audio_direct, ext, num_speakers, triplet_len
    )

    waves = [load_emphas(path, alpha, rate) for path in paths]
    waves = [clear_from_silence(wave) for wave in waves]

    threshold = get_threshold_duration(waves)
    changed_sounds = np.zeros((len(waves), threshold))

    for i, s in enumerate(waves):
        if len(s) > threshold:
            changed_sounds[i] = s[:threshold]
        else:
            changed_sounds[i, :len(s)] = s

    changed_sounds = changed_sounds.reshape((samples_count, 3, -1))
    return changed_sounds, threshold, labels


def segment_spectrogram(stft_data, num_segs, num_ftrs):
    concats = np.concatenate([stft_data[:, 0:num_segs - 1], stft_data], axis=1)
    stft_segs = np.zeros((num_ftrs, num_segs,
                          concats.shape[1] - num_segs + 1))

    for index in range(concats.shape[1] - num_segs + 1):
        stft_segs[:, :, index] = concats[:, index:index + num_segs]

    shape = stft_segs.shape
    stft_segs = np.reshape(stft_segs, (
        shape[0], shape[1], 1, shape[2]
    ))
    stft_segs = np.transpose(
        stft_segs, (3, 0, 1, 2)
    ).astype(np.float32)
    return stft_segs


def get_spectrogram(wave, _nfft, _hop_len, _win_len, _window):
    spec = librosa.stft(wave, n_fft=_nfft, hop_length=_hop_len,
                        win_length=_win_len, window=_window)
    spec = np.abs(spec)
    spec_mean = np.mean(spec)
    spec_std = np.std(spec)
    spec = (spec - spec_mean) / spec_std
    return spec


def get_segmented_spectrograms(audio_direct, ext, num_speakers,
                               triplet_len, alpha, rate):

    sounds, threshold, labels = truncate_or_pad(
        audio_direct, ext, num_speakers, triplet_len, alpha, rate
    )

    spec_shape = (len(sounds), threshold // hop_len + 1,
                  num_features, num_segments, 1)
    anch_specs = np.empty(spec_shape, np.float32)
    pos_specs = np.empty(spec_shape, np.float32)
    neg_specs = np.empty(spec_shape, np.float32)

    for i, three in enumerate(sounds):
        anch_wave, pos_wave, neg_wave = three

        anch_spec = get_spectrogram(anch_wave, nfft, hop_len,
                                    win_len, hamming)
        pos_spec = get_spectrogram(pos_wave, nfft, hop_len,
                                   win_len, hamming)
        neg_spec = get_spectrogram(neg_wave, nfft, hop_len,
                                   win_len, hamming)

        anch_features = segment_spectrogram(
            anch_spec, num_segments, num_features)
        anch_specs[i] = anch_features

        pos_features = segment_spectrogram(
            pos_spec, num_segments, num_features)
        pos_specs[i] = pos_features

        neg_features = segment_spectrogram(
            neg_spec, num_segments, num_features)
        neg_specs[i] = neg_features

    return anch_specs, pos_specs, neg_specs, labels
