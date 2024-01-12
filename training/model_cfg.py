from pathlib import Path

ROOT_DIR=Path('..')
FEATURE_PATH = Path('..', 'features')

N_GPU = 1    # number of GPUs available. Use 0 for CPU mode.
RANDOM_SEED = 42

TRAIN_SIZE = 0.8
VAL_SIZE = 0.1
TEST_SIZE = 0.1
CONTEXT_WINDOW=5





plot_dir = f'{ROOT_DIR}/models/plot'
model_dir = f'{ROOT_DIR}/models/model'
output_dim = 1              # binary problem

lstm_config = {
    'lstm_sf': {
        'input_dim': 25,
        'hidden_dim': 512,
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'model_name': 'lstm_sf.pth',
        'plot_name': 'lstm_sf'
    },
    'lstm_sf_ef': {
        'input_dim': 52,
        'hidden_dim': 512,
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'model_name': 'lstm_sf_ef.pth',
        'plot_name': 'lstm_sf_ef'
    },
    'lstm_sf_ef_ri': {
        'input_dim': 52,
        'hidden_dim': 512,
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'model_name': 'lstm_sf_ef_ri.pth',
        'plot_name': 'lstm_sf_ef_ri'
    },
    'lstm_sf_ef_c': {
        'input_dim': 312,
        'hidden_dim': 512,
        'dropout_rate': 0.1,
        'learning_rate': 1e-3,
        'model_name': 'lstm_sf_ef_c.pth',
        'plot_name': 'lstm_sf_ef_c'
    },
}
