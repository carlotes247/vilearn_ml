from pathlib import Path


DATASET_ROOT_DIR = Path('..', '..', 'wuerzburg_dataset')

FEATURE_ROOT_DIR = Path('..', '..', 'features')

N_GPU = 1  # number of GPUs available. Use 0 for CPU mode.
RANDOM_SEED = 42

#FPS = 58.05
SAMPLING_RATE = 22050  # actual audio: 48.0 kHz
BIT_DEPTH=16           # actual audio: 32 Bit

# audio features
WINDOW_LENGTH=2048
HOP_LENGTH=int(WINDOW_LENGTH*0.25)
N_MFCC = 25
N_FFT = WINDOW_LENGTH

