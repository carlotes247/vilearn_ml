import torch.nn as nn


###################################################
# Define SVM architectures
###################################################


class Bin_SVM(nn.Module):
    def __init__(self, input_dim):
        super(Bin_SVM, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        out = self.linear(x)
        out = out.squeeze()
        return out

