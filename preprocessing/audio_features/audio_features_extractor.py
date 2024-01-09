import pathlib
import librosa
from tqdm import tqdm
from PIL import Image
import numpy as np
from preprocessing.audio_features.audio_cfg import DATASET_ROOT_DIR, WINDOW_LENGTH, HOP_LENGTH, FEATURE_ROOT_DIR
from utils.audio_feature_utils import load_audio, compute_audio_features
from pathlib import Path

def save_audio_features(audio_files, mono:bool, feature_root_path, mfcc_b: bool, prosodic_b: bool, egemaps_b: bool):
    for file in tqdm(audio_files):

        # only extract features for existing files
        if file.is_file():
            audio=load_audio(file, mono)

            # prepare audio for feature concatenation
            audio_channels=[]
            # is audio mono?
            if len(audio.shape)==1:
                audio_channels.append(audio)
            # is audio stereo?
            elif len(audio.shape)==2 and audio.shape[0] == 2:
                audio_channels.append(audio[0])
                audio_channels.append(audio[1])

            # concatenate the features of one channel with the features from other channels
            # resulting in (total_hops_in_audio, feature_size_per_channel * channel_num)
            #features = compute_audio_features(audio, True, True, True)

            concat_f=None
            concat_egemaps_f=None
            for audio_channel in audio_channels:
                features=compute_audio_features(audio_channel, mfcc_b, prosodic_b, egemaps_b)
                # concatenate mfccs, f0, rms of different audio channels, if calculated
                if mfcc_b or prosodic_b:
                    if concat_f is None:
                        concat_f=features[0]
                    else:
                        concat_f=np.concatenate((concat_f, features[0]), axis=1)

                # concatenate egmaps of different audio channels, if calculated
                if egemaps_b:
                    if concat_egemaps_f is None:
                        concat_egemaps_f = features[1]
                    else:
                        concat_egemaps_f = np.concatenate((concat_egemaps_f, features[1]), axis=1)
            print()

            rel_f_path = file.relative_to(DATASET_ROOT_DIR)
            output_folder = Path(feature_root_path,rel_f_path).parent

            if mfcc_b or prosodic_b:
                output_file=Path(output_folder,  str(file.stem) + "_mfcc_f0_rms_wl_" + str(WINDOW_LENGTH) + "_hl_" + str(HOP_LENGTH) + ".npz" )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(output_file, data=concat_f)
            if egemaps_b:
                output_file=Path(output_folder,  str(file.stem) + "_egemaps_wl_" + str(WINDOW_LENGTH) + "_hl_" + str(HOP_LENGTH) + ".npz" )
                output_file.parent.mkdir(parents=True, exist_ok=True)
                np.savez(output_file, data=concat_egemaps_f)


            # np.savez_compressed(output_file.as_posix(), data=overall_features)

            # output_file = pathlib.Path(out_dir, file.stem + '.npz')



# tasks:

#   - Translate: speaker diarization--> "points in time for this feature"

#   - denoise audio (reaper) --> new wav
#   - speaker diarization output && detect speaker manually --> (audio_info.csv)
#   -  filter speaker diarization wrt. speakerX --> speaking_features.npz




#  concat folder names from processed frames with '.wav', resulting in a list of absolute paths for corresponding wavs
audio_files = [file for file in DATASET_ROOT_DIR.glob('**/*.wav')]


#TODO: löschen--->
prefix=Path('..\\..\\wuerzburg_dataset\\wise_2023\\Session 4\\Group 5\\PC_127')
p=Path(prefix, 'test_sample.wav')
audio_files=[p]

audio_files=[audio_files[-1]]
#TODO: <---löschen


save_audio_features(audio_files, True, FEATURE_ROOT_DIR, True, True, True)

print("end of app")



