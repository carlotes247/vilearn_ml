import os
import torch
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.utils.data import DataLoader
from training.architectures.lstm import LSTM
from training.MyDataset import MyDataset
from training.model_cfg import lstm_config, plot_dir, model_dir, output_dim
from training.model_cfg import FEATURE_PATH, ROOT_DIR, N_GPU
from typing import Union




batch_size = 128
epochs = 200



def train_step(neural_network, optimizer: Adam, classification_gt: Union[torch.FloatTensor, torch.Tensor], criterion,
               speech_features: Union[torch.FloatTensor, torch.Tensor]) -> Union[torch.FloatTensor, torch.Tensor]:
    """
    Perform a single training step for the LSTM model.

    Args:
         nn: The nn model.
        optimizer: The Adam optimizer for the nn.
        classification_gt:  A tensor representing the ground truth for the classification.
        speech_features:  A tensor representing the speech features.

    Returns:
        A tensor representing the loss
    """
    lstm_prediction = neural_network(speech_features)

    loss = criterion(lstm_prediction, classification_gt)

    # back propagation and optimization for generator
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def validation_loss(neural_network, val_loader: DataLoader, loss_fn, device) -> float:
    """
    Calculate the validation loss for the LSTM model.

    Args:
        neural_network: The nn model.
        val_loader: The validation data loader.

    Returns:
        A float representing the validation loss.
    """
    neural_network.eval()
    with torch.no_grad():
        num_batches=len(val_loader)
        size=len(val_loader.dataset)
        total_loss=0.0
        correct=0
        for classification_gt, input_features in val_loader:
            classification_gt = classification_gt.to(device)
            speech_features = speech_features.to(device)
            batch_size = classification_gt.shape[0]

            prediction = neural_network(input_features)
            total_loss += loss_fn(prediction, classification_gt).item()
            correct += (prediction.argmax(1) == classification_gt).type(torch.float).sum().item()

    neural_network.train()
    return total_loss / num_batches, correct/size




def main(args):
    spectral_features = args.spectral_features
    emotional_speech_features = args.emotional_speech_features
    remove_identity = args.remove_identity
    add_context = args.context

    if spectral_features and not emotional_speech_features and not add_context:
        model_config = lstm_config['lstm_sf']
    elif spectral_features and emotional_speech_features and not add_context:
        model_config = lstm_config['lstm_sf_ef']
    elif spectral_features and emotional_speech_features and add_context:
        model_config = lstm_config['lstm_sf_ef_c_1']
    else:
        raise Exception('Model config not defined!')

    input_dim = model_config['input_dim']
    hidden_dim = model_config['hidden_dim']
    dropout_rate = model_config['dropout_rate']
    learning_rate = model_config['learning_rate']
    model_name = model_config['model_name']
    plot_name = model_config['plot_name']

    losses = []

    # set up paths
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # set up device
    device = torch.device('cuda:0' if (torch.cuda.is_available() and N_GPU > 0) else 'cpu')
    criterion = nn.MSELoss().to(device)



    # set up  data loader
    training_data = MyDataset(feature_path=FEATURE_PATH, root_dir=ROOT_DIR,
                         spectral_features=spectral_features, emotional_speech_features=emotional_speech_features,
                         add_context=add_context, data_set='train')
    validation_data = MyDataset(feature_path=FEATURE_PATH, root_dir=ROOT_DIR,
                         spectral_features=spectral_features, emotional_speech_features=emotional_speech_features,
                         add_context=add_context, data_set='val')

    data_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, num_workers=2)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, num_workers=2)

    # set up model architecture
    # initialize lstm
    lstm = LSTM(input_dim, hidden_dim, output_dim, dropout_rate).to(device)

    # handle multi-GPU if desired
    if (device.type == 'cuda') and (N_GPU > 1):
        lstm = nn.DataParallel(lstm, list(range(N_GPU)))
    print(f'{lstm}\n')

    # setup Adam optimizers for lstm
    optimizer = Adam(lstm.parameters(), lr=learning_rate)

    lstm.train()
    for epoch in range(epochs):
        # Get your data
        for classification_gt, input_features in data_loader:
            # load batch data
            classification_gt = classification_gt.to(device)
            input_features = input_features.to(device)

            # train lstm
            loss = train_step(lstm, optimizer, criterion, classification_gt, input_features)
            validation_loss(lstm, validation_loader, criterion, device)
            losses.append(loss)

        print(f'Train LSTM Epoch {epoch}, Loss: {loss}')

    # save training
    torch.save(lstm.state_dict(), f'{model_dir}/{model_name}')

    plt.figure(figsize=(10, 5))
    plt.title('LSTM loss during training')
    plt.plot(losses, label='LSTM')
    plt.xlabel('iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{plot_dir}/{plot_name}_loss.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments to the parser
    parser.add_argument('-sf', '--spectral_features', action='store_true', default=True,
                        help='Enable spectral audio features')
    parser.add_argument('-ef', '--emotional_speech_features', action='store_true', default=False,
                        help='Enable emotional speech audio features')
    parser.add_argument('-c', '--context', action='store_true', default=False, help='Add context to trainings data')
    args = parser.parse_args()

    main(args)
