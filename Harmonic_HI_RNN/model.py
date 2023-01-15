import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
from torch import nn

def train_model(data_loader, model, loss_function, optimizer, MLP_flag=False):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        if MLP_flag:
            X = torch.swapaxes(X, 1, 2)
        output = model(X)
        loss = loss_function(output, y.flatten())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    avg_loss = total_loss/num_batches
    return avg_loss

class nts_RNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # hidden_size should be a list with 1-3 elements
        self.hidden_size = hidden_size
        num_hidden_layer = len(hidden_size)
        if num_hidden_layer >= 1:
            self.rnn_1 = nn.RNN(input_size=1, hidden_size=self.hidden_size[0], batch_first=True)
        if num_hidden_layer >= 2:
            self.rnn_2 = nn.RNN(input_size=self.hidden_size[0], hidden_size=self.hidden_size[1])
        if num_hidden_layer == 3:
            self.rnn_3 = nn.RNN(input_size=self.hidden_size[1], hidden_size=self.hidden_size[2])
        self.linear = nn.Linear(in_features=self.hidden_size[num_hidden_layer-1], out_features=1)

    def forward(self, x):
        num_hidden_layer = len(self.hidden_size)

        batch_size = x.shape[0]
        if num_hidden_layer >= 1:
            h1_0 = torch.zeros(1, batch_size, self.hidden_size[0])
        if num_hidden_layer >= 2:
            h2_0 = torch.zeros(1, batch_size, self.hidden_size[1])
        if num_hidden_layer == 3:
            h3_0 = torch.zeros(1, batch_size, self.hidden_size[2])

        if num_hidden_layer == 1:
            _, h1 = self.rnn_1(x, h1_0)
            out = self.linear(h1).flatten()
        elif num_hidden_layer == 2:
            _, h1 = self.rnn_1(x, h1_0)
            _, h2 = self.rnn_2(h1, h2_0)
            out = self.linear(h2).flatten()
        else:
            _, h1 = self.rnn_1(x, h1_0)
            _, h2 = self.rnn_2(h1, h2_0)
            _, h3 = self.rnn_3(h2, h3_0)
            out = self.linear(h3).flatten()
        return out

class nts_MLP(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        # hidden_size should be a list with 1-3 elements
        self.hidden_size = hidden_size
        num_hidden_layer = len(hidden_size)
        if num_hidden_layer >= 1:
            self.fc1 = nn.Linear(in_features=input_size, out_features=self.hidden_size[0])
        if num_hidden_layer >= 2:
            self.fc2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        if num_hidden_layer == 3:
            self.fc3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.linear = nn.Linear(in_features=self.hidden_size[num_hidden_layer-1], out_features=1)

    def forward(self, x):
        num_hidden_layer = len(self.hidden_size)
        F = nn.Tanh()

        if num_hidden_layer == 1:
            h1 = self.fc1(x)
            h1 = F(h1)
            out = self.linear(h1).flatten()
        elif num_hidden_layer == 2:
            h1 = self.fc1(x)
            h1 = F(h1)
            h2 = self.fc2(h1)
            h2 = F(h2)
            out = self.linear(h2).flatten()
        else:
            h1 = self.fc1(x)
            h1 = F(h1)
            h2 = self.fc2(h1)
            h2 = F(h2)
            h3 = self.fc3(h2)
            h3 = F(h3)
            out = self.linear(h3).flatten()
        return out