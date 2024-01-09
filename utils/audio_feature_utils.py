from tqdm import tqdm
import copy
import opensmile
import librosa
import warnings
import numpy as np
from typing import Tuple
from preprocessing.audio_features.audio_cfg import DATASET_ROOT_DIR,  SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH, N_MFCC, N_FFT


def load_audio(audio_file: str, mono: bool) -> np.ndarray:
    """
    Load an audio file and resample it to a standard sampling rate.

    Args:
        audio_file: A string indicating the path to the audio file.

    Returns:
        A numpy array containing the loaded and resampled audio.
    """
    # load audio
    audio, sr = librosa.load(audio_file, sr=None, mono=mono)
    # standardize sampling rate
    audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    return audio



def compute_audio_features(audio: np.ndarray, mfcc_b: bool, prosodic_b: bool, egemaps_b: bool) -> np.ndarray:
    """
    Computes audio features based on the given audio file and feature options.
    These are the features concatenated per channel with the size (total_hops_per_audio, featuresize)

    Args:
        audio: A numpy array containing the audio sequence.
        mfcc_b: A boolean flag indicating whether to include spectral features (MFCCs).
        prosocic_b: A boolean flag indicating whether to include prosodic features (fundamental frequence intensity, rms).
        emotional_speech_features: A boolean flag indicating whether to include emotional speech features.

    Returns:
        A numpy array of computed audio features.
    """
    mfccs=None
    prosodic=None
    egemaps=None

    if mfcc_b:
        mfccs = extract_spectral_features(audio,  SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH, N_MFCC, N_FFT)
    if prosodic_b:
        prosodic = extract_fundamental_freq_intensity(audio, SAMPLING_RATE, WINDOW_LENGTH, HOP_LENGTH)
        prosodic=np.concatenate((prosodic[0], prosodic[1]), axis=1)
        print()

    if egemaps_b:
       egemaps = extract_emotional_speech_features(audio)

        # # cut mfcc, prosodic features to size of egemaps
        # if mfcc_b:
        #     diff_f_idx = len(mfccs) - len(egemaps)
        #     mfccs=mfccs[diff_f_idx:]
        # if prosodic_b:
        #     diff_f_idx = len(prosodic) - len(egemaps)
        #     prosodic=prosodic[diff_f_idx:]

    # concat features to one np array of shape (total_hops_in_audio, 25+25+25)
    # features_to_concat = []
    # if mfcc_b:
    #     features_to_concat.append(mfccs)
    # if prosodic_b:
    #     features_to_concat.append(prosodic)
    # if egemaps_b:
    #     features_to_concat.append(egemaps)

    feature_out=np.concatenate((mfccs, prosodic), axis=1)

    return feature_out, egemaps


def extract_spectral_features(audio: np.ndarray, sr, window_length, hop_length, n_mfcc, n_fft) -> np.ndarray:
    """
    Extracts spectral features (MFCCs) from an audio file.

    Args:
        audio: A numpy array containing the audio sequence.

    Returns:
        A numpy array of MFCCs (Mel-frequency cepstral coefficients).
    """
    mel_spec = librosa.feature.mfcc(y=audio, sr=sr, win_length=window_length,  hop_length=hop_length, n_mfcc=n_mfcc, n_fft=n_fft)
    mel_spec = mel_spec.transpose()
    return mel_spec


def extract_fundamental_freq_intensity(audio, sr, window_length, hop_length) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts fundamental frequency (F0) and intensity (energy) features from an audio file.
    Args:
        audio: A numpy array containing the audio sequence.

    Returns:
        Tuple containing the F0 array and intensity array.
    """

    fmin = librosa.note_to_hz('C2')
    fmax = librosa.note_to_hz('C7')

    f0, voiced_flag, voiced_probs = librosa.pyin(y=audio, fmin=fmin, fmax=fmax, sr=sr, frame_length=window_length,
                                                 hop_length=hop_length)
    f0 = np.nan_to_num(f0)

    # librosa doesn't have a direct function to compute the intensity (or loudness) of an audio signal.
    # One commonly used measure is the root mean square (RMS) energy of the signal
    rms = librosa.feature.rms(y=audio, frame_length=window_length, hop_length=hop_length)[0]

    return f0.reshape(-1,1), rms.reshape(-1,1)


def extract_emotional_speech_features(audio) -> np.ndarray:
    """
     Extracts emotional speech features from an audio file using OpenSMILE.

    Args:
        audio: A numpy array containing the audio sequence.

    Returns:
        A numpy array of emotional speech features
    """
    # frameSize = 0.09287981859410431
    # frameStep = 0.023219954648526078
    # venv/Lib/site-packages/opensmile/core/config/gemaps/v01b/GeMAPSv01b_core.lld.conf.inc
    # frameSize = 0.09287981859410431
    # frameStep = 0.023219954648526078

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )

    eGeMAPS_features = smile.process_signal(audio, SAMPLING_RATE)
    return eGeMAPS_features.to_numpy()



