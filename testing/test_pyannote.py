import numpy as np




token='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'

# dataset:
# https://oc.informatik.uni-wuerzburg.de/apps/files/?dir=/VILEARN%20MORE/Augsburg_Builds_ViLearn/Recordings/2023_12_19_Seminar_Wue&fileid=45579884
#Test: 2023_12_19_Seminar_Wue\Session 1\Group 4\PC_128
# important: also the slight backchannels (noise gate shouldnt be too restrictive)

################noise gate################
# Alternative1:
# https://github.com/timsainb/noisereduce/tree/master
# https://pypi.org/project/noisereduce/

# Alternative2: sox can be configured for noise gate: https://stackoverflow.com/questions/18985952/sox-how-to-noise-gate

# Alternative3: routing of nvidia broadcast, but then begin has to be synced
# https://www.youtube.com/watch?v=Z3QeaXhfkGg

################noise gate################

################VAD################
'''
Pyannote repo: https://github.com/pyannote/pyannote-audio
From there, pipelines: https://huggingface.co/models?other=pyannote-audio-pipeline
From there, training: https://huggingface.co/models?other=pyannote-audio-model

Current VAD Model used: https://huggingface.co/pyannote/voice-activity-detection
'''
################VAD################




#audio_file='audio_own'
audio_file='audio_vilearn'

audio_file_channeled=audio_file + "channel1"
audio_file_gated=audio_file_channeled + "gated"
'''
from scipy.io import wavfile
import noisereduce as nr
from pydub import AudioSegment

aud_seg = AudioSegment.from_wav(audio_file + ".wav")
aud_seg = aud_seg.set_channels(1)
aud_seg.export(audio_file_channeled + '.wav', format="wav")


# load data
rate, data = wavfile.read(audio_file_channeled + '.wav')
# perform noise reduction
reduced_noise = nr.reduce_noise(y = data, sr=rate, stationary=False, prop_decrease=0.7)
wavfile.write(audio_file_gated + ".wav", rate, reduced_noise)
print()

rate, data = wavfile.read(audio_file_gated + '.wav')
# perform noise reduction
reduced_noise = nr.reduce_noise(y = data, sr=rate, stationary=False)
wavfile.write(audio_file_gated + "2.wav", rate, reduced_noise)
print()


'''


'''
# VAD PIPELINE
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=token)
output = pipeline(audio_file)

for speech in output.get_timeline().support():
    print(str(speech))
'''


''' '''

from datetime import timedelta
from pathlib import Path


elan_csv_header=[
    'Tier',
    'Begin',
    'Begin_no',
    'End',
    'End_no',
    'Duration',
    'Duration_no',
    'Annotation'
]





'''
# test convertTimeToElan(ts)


a = convertTimeToElan(3601.340)
b = convertTimeToElan(3661.803)
c = convertTimeToElan(3600.810)
d = convertTimeToElan(3559.0001)
print()
'''
def convertTimeToElan(ts: float):
    # Calculate hours, minutes, seconds, and milliseconds
    hours, remainder = divmod(ts, 3600)
    minutes, remainder2 = divmod(remainder, 60)

    milliseconds, seconds = int (round(remainder2 - int(remainder2), 3)*1000 ) , int(remainder2)


    # Create a timedelta object
    time_delta = timedelta(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)

    # Manually format the milliseconds part
    milliseconds_str = "{:03d}".format(time_delta.microseconds // 1000)

    # Format the timedelta as a string in the desired format
    time_format = "{:02d}:{:02d}:{:02d}.{}".format(int(hours), int(minutes), int(seconds), milliseconds_str)

    return time_format

def speaker_diarization_to_csv(file_path:Path, line_separator:str, csv_header:[], csv_rows:[[]]):
    import csv

    # Writing to CSV file
    with open(file_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=line_separator)

        # Write the header
        csv_writer.writerow(csv_header)

        # Write the data row
        for row in csv_rows:
            csv_writer.writerow(row)

def predict_speakers(audio_file_path:Path, file_path:Path, csv_header, line_separator, save:bool, output:bool, tier:str):
    # SPEAKER DIARIZATION PIPELINE
    import torch, torchaudio
    from pyannote.audio import Pipeline
    from pyannote.audio.pipelines.utils.hook import ProgressHook

    pipeline = Pipeline.from_pretrained(
      "pyannote/speaker-diarization-3.1",
      use_auth_token=token)

    '''
    
    '''
    if torch.cuda.is_available():
        dev='cuda'
    else:
        dev='cpu'
    
    # send pipeline to GPU (when available)
    pipeline.to(torch.device(dev))
    
    #apply pretrained pipeline
    waveform, sample_rate = torchaudio.load(audio_file + ".wav")
    
    audio=waveform.reshape(1,-1)
    with ProgressHook() as hook:
        # run the pipeline on an audio file
        audio_p = str(audio_file_path)
        diarization = pipeline(audio_p, hook=hook)
        #diarization = pipeline({"waveform": audio, "sample_rate": sample_rate, "hook": hook})


    #diarization = pipeline(audio_p)
    csv_rows=[]
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if output:
            print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")
        if save:
            start_str=convertTimeToElan(turn.start)
            end_str=convertTimeToElan(turn.end)
            dur_str=convertTimeToElan(turn.duration)
            row=[
                tier,
                start_str,
                str(round(turn.start, 2)),
                end_str,
                str(round(turn.end, 2)),
                dur_str,
                str(round(turn.duration, 2)),
                str(speaker)
            ]
            csv_rows.append(row)
            print()

    if save:
        speaker_diarization_to_csv( file_path, line_separator, csv_header, csv_rows)


def predict_speakers_vad(audio_file_path:Path, file_path:Path, csv_header, line_separator, save:bool, output:bool, tier:str, accuracy=10):
    # https://github.com/MorenoLaQuatra/vad

    from vad.energy_vad import EnergyVAD
    import torch
    import torchaudio
    import numpy as np
    import matplotlib.pyplot as plt

    audio=str(audio_file_path)
    clean_filename = audio.split("/")[-1].split(".")[0]

    # Load audio
    waveform, sample_rate = torchaudio.load(audio)


    fl=waveform.shape[1]/sample_rate*1000
    fl2=fl * sample_rate // 1000

    # Compute VAD
    vad = EnergyVAD(
        sample_rate=sample_rate,
        frame_length=int(fl),
        #frame_shift=0.022675736961451247,
        energy_threshold=0.05,
        pre_emphasis=0.95
    )
    vad_output = vad(waveform.numpy())

    if save:
        # collect list of (start,end)-index tuples referring to index in numpy array
        previous=-1.0
        start=-1
        all_times=[]
        for i, value in enumerate(vad_output):
            if previous==0.0 and value==0.0:
                continue
            elif previous==1.0 and value==1.0:
                continue
            elif previous==0.0 and value==1.0:
                start=i
            elif (previous==1.0 and value==0.0) or (previous==1.0 and i==len(vad_output)-1):
                end=i-1
                all_times.append((start,end))

        # calc actual times on ms basis as float of each sample in the audio
        y_range = np.arange(len(waveform))
        y_times = librosa.frames_to_time(y_range, sr=sample_rate, hop_length=1)

        csv_rows=[]
        for start, end in all_times:
            new_start, new_end = y_times[start], y_times[end]   # using indices (start,end) crop the times out of y_times--> new_start, new_end
            start_str = convertTimeToElan(new_start)
            end_str = convertTimeToElan(new_end)
            dur_str = convertTimeToElan(new_end-new_start)
            row = [
                tier,
                start_str,
                str(round(new_start, accuracy)),
                end_str,
                str(round(new_end, accuracy)),
                dur_str,
                str(round(new_end-new_start, accuracy)),
                str('')
            ]
            csv_rows.append(row)
            print()

        speaker_diarization_to_csv( file_path, line_separator, csv_header, csv_rows)






import contextlib
import sys
import wave

import webrtcvad


def read_wave(path):
    """
    from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
    Reads a .wav file.


    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


import collections
'''
    https://github.com/wiseman/py-webrtcvad/blob/master/example.py
'''


class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.

    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.

    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.

    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.

    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.

    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.

    Arguments:

    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).

    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])



'''
from https://github.com/wiseman/py-webrtcvad/blob/master/example.py
'''
def predict_speakers_vad_2(audio_file_path: Path, file_path: Path, csv_header, line_separator, save: bool,
                         output: bool, tier: str, accuracy=10):
    audio = str(audio_file_path.as_posix())

    # # Load the audio file as mono and resample it to 16 kHz
    # y, sr = librosa.load(audio, sr=16000, mono=True)
    #
    # # Save the resampled audio to the temporary file
    # librosa.output.write_wav('tmp.wav', y, sr)
    # audio='tmp.wav'

    audio, sample_rate = read_wave(audio)
    vad = webrtcvad.Vad(int(0))
    frames = frame_generator(30, audio, sample_rate)
    frames = list(frames)

    print()









import librosa
audio = str( Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.wav') )

predict_speakers_vad_2(Path('example_mono.wav'),
                 Path('example_mono.csv'),
                 elan_csv_header,
                 ';',
                 save=True,
                 output=True,
                 tier='speaker_tier')

# predict_speakers_vad(Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.wav'),
#                  Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.csv'),
#                  elan_csv_header,
#                  ';',
#                  save=True,
#                  output=True,
#                  tier='speaker_tier')

print()


'''

predict_speakers(Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.wav'),
                 Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.csv'),
                 elan_csv_header,
                 ';',
                 save=True,
                 output=True,
                 tier='speaker_tier')




'''
#predict_speakers(Path('DESKTOP-QTU96C2_vilearn2023-10-30__15-08-32.921.wav'), Path('DESKTOP-QTU96C2_vilearn2023-10-30__15-08-32.921' + '.csv'), elan_csv_header, ';', True, True, tier='speaker_tier')
#predict_speakers(Path('DESKTOP-VK97U75_vilearn2023-10-30__15-08-32.173.wav'), Path('DESKTOP-VK97U75_vilearn2023-10-30__15-08-32.173' + '.csv'), elan_csv_header, ';', True, True, tier='speaker_tier')

'''
# SPEAKER SEGMENTATION PIPELINE
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation", use_auth_token=token)
output = pipeline(audio_file + ".wav")

for turn, _, speaker in output.itertracks(yield_label=True):
    print('turn: ' + str(turn) + ' _: ' + str(_) + ' speaker: ' + str(speaker))
'''
