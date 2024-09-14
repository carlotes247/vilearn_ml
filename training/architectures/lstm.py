import torch
import torch.nn as nn


###################################################
# Define LSTM architectures
###################################################


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(LSTM, self).__init__()
        self.num_layers = 4
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=self.num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        x = x.unsqueeze(1)
        #print(f"x shape: {x.shape}")
        #print(f"h0 shape: {h0.shape}")
        #print(f"c0 shape: {c0.shape}")
        out, _ = self.lstm(x, (h0, c0))
        out = self.sigmoid(out)
        out = self.fc(out)

        return out

