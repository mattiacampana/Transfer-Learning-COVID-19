import json
import os
import sys
import warnings
import openl3
import librosa

import numpy as np
import pandas as pd

from scipy.fftpack import fft, hilbert
from math import pi
from tqdm import tqdm

warnings.filterwarnings("ignore")

SR = 22050 # sample rate
sr = SR
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
MFCC_dim = 13  # the MFCC dimension

def sta_fun(np_data):
    """Extract various statistical features from the numpy array provided as input.
    :param np_data: the numpy array to extract the features from
    :type np_data: numpy.ndarray
    :return: The extracted features as a vector
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if np_data is None:
        raise ValueError("Input array cannot be None")

    # perform the feature extraction
    dat_min = np.min(np_data)
    dat_max = np.max(np_data)
    dat_mean = np.mean(np_data)
    dat_rms = np.sqrt(np.sum(np.square(np_data)) / len(np_data))
    dat_median = np.median(np_data)
    dat_qrl1 = np.percentile(np_data, 25)
    dat_qrl3 = np.percentile(np_data, 75)
    dat_lower_q = np.quantile(np_data, 0.25 , interpolation = "lower")
    dat_higher_q = np.quantile(np_data, 0.75, interpolation = "higher")
    dat_iqrl = dat_higher_q - dat_lower_q
    dat_std = np.std(np_data)
    s = pd.Series(np_data)
    dat_skew = s.skew()
    dat_kurt = s.kurt()

    # finally return the features in a concatenated array (as a vector)
    return np.array([dat_mean, dat_min, dat_max, dat_std, dat_rms,
                     dat_median, dat_qrl1, dat_qrl3, dat_iqrl, dat_skew, dat_kurt])

def get_period(signal, signal_sr):
    """Extract the period from the the provided signal
    :param signal: the signal to extract the period from
    :type signal: numpy.ndarray
    :param signal_sr: the sampling rate of the input signal
    :type signal_sr: integer
    :return: a vector containing the signal period
    :rtype: numpy.ndarray
    """

    # perform a sanity check
    if signal is None:
        raise ValueError("Input signal cannot be None")

    # transform the signal to the hilbert space
    hy = hilbert(signal)

    ey = np.sqrt(signal ** 2 + hy ** 2)
    min_time = 1.0 / signal_sr
    tot_time = len(ey) * min_time
    pow_ft = np.abs(fft(ey))
    peak_freq = pow_ft[3: int(len(pow_ft) / 2)]
    peak_freq_pos = peak_freq.argmax()
    peak_freq_val = 2 * pi * (peak_freq_pos + 2) / tot_time
    period = 2 * pi / peak_freq_val

    return np.array([period])

def extract_signal_features(signal, signal_sr):
    """Extract part of handcrafted features from the input signal.
    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """

    # normalise the sound signal before processing
    signal = signal / np.max(np.abs(signal))
    # trim the signal to the appropriate length
    trimmed_signal, idc = librosa.effects.trim(signal, frame_length=FRAME_LEN, hop_length=HOP)
    # extract the signal duration
    signal_duration = librosa.get_duration(y=trimmed_signal, sr=signal_sr)
    # use librosa to track the beats
    tempo, beats = librosa.beat.beat_track(y=trimmed_signal, sr=signal_sr)
    # find the onset strength of the trimmed signal
    o_env = librosa.onset.onset_strength(trimmed_signal, sr=signal_sr)
    # find the frames of the onset
    onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=signal_sr)
    # keep only the first onset frame
    onsets = onset_frames.shape[0]
    # decompose the signal into its magnitude and the phase components such that signal = mag * phase
    mag, phase = librosa.magphase(librosa.stft(trimmed_signal, n_fft=FRAME_LEN, hop_length=HOP))
    # extract the rms from the magnitude component
    rms = librosa.feature.rms(y=trimmed_signal)[0]
    # extract the spectral centroid of the magnitude
    cent = librosa.feature.spectral_centroid(S=mag)[0]
    # extract the spectral rolloff point from the magnitude
    rolloff = librosa.feature.spectral_rolloff(S=mag, sr=signal_sr)[0]
    # extract the zero crossing rate from the trimmed signal using the predefined frame and hop lengths
    zcr = librosa.feature.zero_crossing_rate(trimmed_signal, frame_length=FRAME_LEN, hop_length=HOP)[0]

    # pack the extracted features into the feature vector to be returned
    signal_features = np.concatenate(
        (
            np.array([signal_duration, tempo, onsets]),
            get_period(signal, signal_sr=sr),
            sta_fun(rms),
            sta_fun(cent),
            sta_fun(rolloff),
            sta_fun(zcr),
        ),
        axis=0,
    )

    # finally, return the gathered features and the trimmed signal
    return signal_features, trimmed_signal

def extract_mfcc(signal, signal_sr=SR, n_fft=FRAME_LEN, hop_length=HOP, n_mfcc=MFCC_dim):
    """Extracts the Mel-frequency cepstral coefficients (MFCC) from the provided signal
    :param signal: the signal to extract the mfcc from
    :type signal: numpy.ndarray
    :param signal_sr: the signal sample rate
    :type signal_sr: integer
    :param n_fft: the fft window size
    :type n_fft: integer
    :param hop_length: the hop length
    :type hop_length: integer
    :param n_mfcc: the dimension of the mfcc
    :type n_mfcc: integer
    :return: the populated feature vector
    :rtype: numpy.ndarray
    """
    # compute the mfcc of the input signal
    mfcc = librosa.feature.mfcc(
        y = signal, sr = signal_sr, n_fft = n_fft, hop_length = hop_length, n_mfcc = n_mfcc, dct_type = 3
    )

    # extract the first and second order deltas from the retrieved mfcc's
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # create the mfcc array
    mfccs = []

    # populate it using the extracted features
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc[i, :])) 
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta[i, :]))
    for i in range(n_mfcc):
        mfccs.extend(sta_fun(mfcc_delta2[i, :]))

    # finally return the coefficients
    return mfccs

def extract_manual_features(file, sr): 
    """Extract all features from the input signal. 
    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the extracted feature vector
    :rtype: numpy.ndarray
    """
    signal, signal_sr = librosa.load(file, sr=sr)

    # extract the signal features
    signal_features, trimmed_signal = extract_signal_features(signal, signal_sr)

    # extract the mfcc's from the trimmed signal and get the statistical feature. 
    mfccs = extract_mfcc(trimmed_signal)

    return np.concatenate((signal_features, mfccs), axis=0)


def extract_manual_features_from_waveform(waveform, sr): 
    """Extract all features from the input signal. 
    :param signal: the signal the extract features from
    :type signal: numpy.ndarray
    :param signal_sr: the sample rate of the signal
    :type signal_sr: integer
    :return: the extracted feature vector
    :rtype: numpy.ndarray
    """
    # extract the signal features
    signal_features, trimmed_signal = extract_signal_features(waveform, sr)

    # extract the mfcc's from the trimmed signal and get the statistical feature. 
    mfccs = extract_mfcc(trimmed_signal)

    return np.concatenate((signal_features, mfccs), axis=0)


def get_resort(files):
    """Re-sort the files under data path.
    :param files: file list
    :type files: list
    :return: alphabetic orders
    :rtype: list
    """
    name_dict = {}
    for sample in files:
        type, name, others = sample.split('_',2)  # the UID is a mixed of upper and lower characters
        name = name.lower()
        name_dict['_'.join([type, name, others])] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    return re_file