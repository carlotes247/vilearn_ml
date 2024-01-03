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
################noise gate################

################VAD################
'''
Pyannote repo: https://github.com/pyannote/pyannote-audio
From there, pipelines: https://huggingface.co/models?other=pyannote-audio-pipeline
From there, models: https://huggingface.co/models?other=pyannote-audio-model

Current VAD Model used: https://huggingface.co/pyannote/voice-activity-detection
'''
################VAD################




audio_file='audio_own'
audio_file='audio_vilearn'

audio_file_channeled=audio_file + "channel1"
audio_file_gated=audio_file_channeled + "gated"

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



from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained("pyannote/voice-activity-detection",
                                    use_auth_token=token)
output = pipeline(audio_file)

for speech in output.get_timeline().support():
    print(str(speech))
'''


'''
'''