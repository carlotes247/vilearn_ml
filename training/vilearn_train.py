from training.architectures.lstm import LSTM
from training.architectures.svm import Bin_SVM
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from torch_vilearn.torch_group_data_loader import TorchGroupDataLoader
import numpy as np

class ViLearnTrainLogic():
    model: nn.Module
    input_dim: int 
    hidden_dim: int 
    output_dim: int 
    dropout_rate: float 
    learning_rate: float
    device: str 
    # training params
    epochs: int 
    loss_function_criterion: nn.modules.loss._Loss 
    computed_loss_training: nn.modules.loss._Loss 
    computed_mse_loss_eval: nn.modules.loss._Loss
    computed_rmse_loss_eval: nn.modules.loss._Loss
    optimizer: Adam

    def __init__(self) -> None:
        # set up model architecture
        # initialize lstm
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.input_dim = 9
        self.hidden_dim = 8
        self.output_dim = 1 # 1 is binary, but in our case it would be a regression so I need to change this
        self.dropout_rate = 0.1    
        self.learning_rate = 1e-3
        self.loss_function_criterion = nn.MSELoss().to(self.device)
        self.model = LSTM(self.input_dim, self.hidden_dim, self.output_dim, self.dropout_rate).to(self.device)
         # setup Adam optimizers for lstm
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        # num epochs
        self.epochs = 10

    def train_lstm(self, train_data_loader: DataLoader, eval_data_loader: DataLoader) -> None:
        
        for epoch in range(self.epochs):
            # Iterate through data batches returned by dataloader
            for input_features, cognition_label in train_data_loader:                
                # load batch data
                features_batch = input_features.to(self.device)
                label_batch = cognition_label.to(self.device)  
                ## TRAINING STEP
                # set lstm to train mode
                self.model.train()
                # Forward pass
                outputs = self.model(features_batch)
                # Compute loss
                self.computed_loss_training = self.loss_function_criterion(outputs, label_batch)
                # Backward pass (backprop) and optimization
                self.optimizer.zero_grad()
                self.computed_loss_training.backward()
                self.optimizer.step()

            ## EVALUATION STEP
            # set lstm to eval mode
            self.model.eval()
            mse_losses = []
            rmse_losses = []
            with torch.no_grad():
                for input_features, cognition_label in eval_data_loader:
                    # load batch data
                    features_batch = input_features.to(self.device)
                    label_batch = cognition_label.to(self.device)  
                    # inference
                    outputs = self.model(features_batch)
                    # Compute losses, both mse and rmse
                    # MSE
                    self.computed_mse_loss_eval = self.loss_function_criterion(outputs, label_batch) 
                    mse_losses.append(self.computed_mse_loss_eval.item())
                    # RMSE
                    self.computed_rmse_loss_eval = torch.sqrt(self.computed_loss_training)
                    rmse_losses.append(self.computed_rmse_loss_eval.item())

            # print average mse and rmse after evaluation loop                
            print(f'Epoch [{epoch+1}/{self.epochs}], RMSE: {np.average(rmse_losses)}, MSE: {np.average(mse_losses)}')


                