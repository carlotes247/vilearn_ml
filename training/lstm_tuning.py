import torch
import optuna
import random
import argparse
import torch.nn as nn
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from training.MyDataset import MyDataset
from training.architectures.lstm import LSTM
from training.lstm_train import batch_size, epochs, output_dim
from training.model_cfg import lstm_config, RANDOM_SEED,N_GPU
from training.model_cfg import FEATURE_PATH, ROOT_DIR
from training.lstm_train import train_step, validation_loss


# set random seed
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

parser = argparse.ArgumentParser()
# Add arguments to the parser
parser.add_argument('-sf', '--spectral_features', action='store_true', default=True,
                    help='Enable spectral audio features')
parser.add_argument('-ef', '--emotional_speech_features', action='store_true', default=False,
                    help='Enable emotional speech audio features')
parser.add_argument('-ri', '--remove_identity', action='store_true', default=False,
                    help='Remove identity information from landmarks')
parser.add_argument('-c', '--context', action='store_true', default=False, help='Add context to trainings data')
args = parser.parse_args()

spectral_features = args.spectral_features
emotional_speech_features = args.emotional_speech_features
remove_identity = args.remove_identity
add_context = args.context

if spectral_features and not emotional_speech_features and not remove_identity and not add_context:
    model_config = lstm_config['lstm_sf']
elif spectral_features and emotional_speech_features and not remove_identity and not add_context:
    model_config = lstm_config['lstm_sf_ef']
elif spectral_features and emotional_speech_features and remove_identity and not add_context:
    model_config = lstm_config['lstm_sf_ef_ri']
elif spectral_features and emotional_speech_features and not remove_identity and add_context:
    model_config = lstm_config['lstm_sf_ef_c']
else:
    raise Exception('Model config not defined!')

input_dim = model_config['input_dim']




# set up  data loaders
training_data = MyDataset(feature_path=FEATURE_PATH, root_dir=ROOT_DIR,
                          spectral_features=spectral_features, emotional_speech_features=emotional_speech_features,
                          add_context=add_context, data_set='train')
train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=0)

# do not remove identity information from validation set
validation_data = MyDataset(feature_path=FEATURE_PATH, root_dir=ROOT_DIR,
                            spectral_features=spectral_features, emotional_speech_features=emotional_speech_features,
                            add_context=add_context, data_set='val')
val_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=0)

# set up device
device = torch.device('cuda:0' if (torch.cuda.is_available() and N_GPU > 0) else 'cpu')


def objective(trial):
    # The following trial covers more cases, and it would be better to test them,
    # but due to the low computation power the computation time is too long
    # hidden_dim = trial.suggest_int('hidden_dim', 16, 512)
    # dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    # learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)

    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256, 512])
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = trial.suggest_categorical("learning_rate", [1e-5, 1e-4, 1e-3])

    # set up model architecture
    # initialize lstm
    lstm = LSTM(input_dim, hidden_dim, output_dim, dropout_rate).to(device)
    # handle multi-GPU if desired
    if (device.type == 'cuda') and (N_GPU > 1):
        lstm = nn.DataParallel(lstm, list(range(N_GPU)))
    print(f'{lstm}\n')

    optimizer = Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(epochs // 2):
        for lip_movements, speech_features in train_loader:
            # load batch data
            lip_movements = lip_movements.to(device)
            speech_features = speech_features.to(device)

            loss = train_step(lstm, optimizer, lip_movements, speech_features)

        # early stopping based on validation loss
        val_loss = validation_loss(lstm, val_loader)
        trial.report(val_loss, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return validation_loss(lstm, val_loader)


def main():
    study = optuna.create_study(study_name='lstm_tuning', direction='minimize')
    study.optimize(objective, n_trials=30)  # n_trials=100

    # print the best hyper parameters
    print(study.best_trial.params)


if __name__ == "__main__":
    main()
