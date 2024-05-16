from architectures.lstm import LSTM
from architectures.svm import Bin_SVM
import torch.nn as nn
import torch
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam
from data_reading.torch_vilearn.torch_group_data_loader import TorchGroupDataLoader

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
    computed_loss: nn.modules.loss._Loss 
    optimizer: Adam

    def __init__(self) -> None:
        # set up model architecture
        # initialize lstm
        self.device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 1 # 1 is binary, but in our case it would be a regression so I need to change this
        self.dropout_rate = 0.1    
        self.learning_rate = 1e-3
        self.loss_function_criterion = nn.MSELoss().to(self.device)
        self.model = LSTM(self.input_dim, self.hidden_dim, self.output_dim, self.dropout_rate).to(self.device)
         # setup Adam optimizers for lstm
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

    def train_lstm(self, data_loader: DataLoader) -> None:
        # set lstm to train mode
        self.model.train()
        for epoch in range(self.epochs):
            # Iterate through data batches returned by dataloader
            for input_features, cognition_label in data_loader:
                # load batch data
                features_batch = input_features.to(self.device)
                label_batch = cognition_label.to(self.device)  
                ## TRAINING STEP
                # Forward pass
                outputs = self.model(features_batch)
                # Compute loss
                self.computed_loss = self.loss_function_criterion(outputs, label_batch)
                # Backward pass (backprop) and optimization
                self.optimizer.zero_grad()
                self.computed_loss.backward()
                self.optimizer.step()

                ## EVALUATION STEP
                

            print(f'Epoch {epoch+1}/{self.epochs}, Loss: {self.computed_loss.item()}')

                