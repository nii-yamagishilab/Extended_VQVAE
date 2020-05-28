# ==============================================================================
# Copyright (c) 2020, Yamagishi Laboratory, National Institute of Informatics
# Original: https://github.com/mkotha/WaveRNN
# Modified: Yi Zhao (zhaoyi[at]nii.ac.jp)
# All rights reserved.
# ==============================================================================
import numpy as np
import librosa, math
import scipy
import config
# sample_rate = 22050
# n_fft = 2048
# fft_bins = n_fft // 2 + 1
# num_mels = 80
# hop_length = 64
# win_length = 1024
# fmin = 40
# min_level_db = -100
# ref_level_db = 20

sample_rate = config.sample_rate
n_fft = config.n_fft
fft_bins = config.fft_bins
num_mels = config.num_mels
hop_length = config.hop_length
win_length = config.win_length
fmin = config.fmin
fmax = config.fmax
min_level_db = config.min_level_db
ref_level_db = config.ref_level_db
print (sample_rate, n_fft, num_mels, hop_length, win_length)

def load_wav(filename, encode=True) :
    x = librosa.load(filename, sr=sample_rate)[0]
    if encode == True : x = encode_16bits(x)
    return x

def save_wav(y, filename) :
    if y.dtype != 'int16' :
        y = encode_16bits(y)
    #librosa.output.write_wav(filename, y.astype(np.int16), sample_rate)
    scipy.io.wavfile.write(filename, sample_rate, y.astype(np.int16))

def split_signal(x) :
    unsigned = x + 2**15
    coarse = unsigned // 256
    fine = unsigned % 256
    return coarse, fine

def combine_signal(coarse, fine) :
    return coarse * 256 + fine - 2**15

def encode_16bits(x) :
    return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

mel_basis = None

def linear_to_mel(spectrogram):
    global mel_basis
    if mel_basis is None:
        mel_basis = build_mel_basis()
    return np.dot(mel_basis, spectrogram)

def build_mel_basis():
    return librosa.filters.mel(sample_rate, n_fft, n_mels=num_mels, fmin=fmin)

def normalize(S):
    return np.clip((S - min_level_db) / -min_level_db, 0, 1)

def denormalize(S):
    return (np.clip(S, 0, 1) * -min_level_db) + min_level_db

def amp_to_db(x):
    return 20 * np.log10(np.maximum(1e-5, x))

def db_to_amp(x):
    return np.power(10.0, x * 0.05)

def spectrogram(y):
    D = stft(y)
    S = amp_to_db(np.abs(D)) - ref_level_db
    return normalize(S)

def melspectrogram(y):
    D = stft(y)
    S = amp_to_db(linear_to_mel(np.abs(D)))
    return normalize(S)

def stft(y):
    return librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
