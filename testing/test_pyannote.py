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
        diarization = pipeline({"waveform": audio, "sample_rate": sample_rate, "hook":hook})
    '''
    # run the pipeline on an audio file
    diarization = pipeline(str(audio_file_path))
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




predict_speakers(Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.wav'), Path('DESKTOP-IV41AK1_vilearn2023-10-30__15-08-39.270.csv'), elan_csv_header, ';', True, True, tier='speaker_tier')
predict_speakers(Path('DESKTOP-QTU96C2_vilearn2023-10-30__15-08-32.921.wav'), Path('DESKTOP-QTU96C2_vilearn2023-10-30__15-08-32.921' + '.csv'), elan_csv_header, ';', True, True, tier='speaker_tier')
predict_speakers(Path('DESKTOP-VK97U75_vilearn2023-10-30__15-08-32.173.wav'), Path('DESKTOP-VK97U75_vilearn2023-10-30__15-08-32.173' + '.csv'), elan_csv_header, ';', True, True, tier='speaker_tier')

'''
# SPEAKER SEGMENTATION PIPELINE
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-segmentation", use_auth_token=token)
output = pipeline(audio_file + ".wav")

for turn, _, speaker in output.itertracks(yield_label=True):
    print('turn: ' + str(turn) + ' _: ' + str(_) + ' speaker: ' + str(speaker))
'''
