The audio features per participant are currently calculated and concatenated as follows (resulting in a array of shape (total_hops_in_audio, 25+2+25) ):

- audio (sr=48000, bit_depth=32) is resampled using the sample rate=22050
- we calculate (using window_length=2048, hop_length=window_length*0.25=512 (librosa standard)):
* 25 mfcc-features 
* f0, rms
* 25 egemaps-features (window_length= 2048/22050, hop_length=window_length*0.25)
