import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

df = pd.read_csv("/home/b3njah/Downloads/df.csv", usecols=['Adj Close'], dtype=np.float32)

input_size = 1
sequence_length = 1
num_layers = 2
hidden_size = 128


class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__(input_size, hidden_size, num_layers)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        return self.rnn(x)


model = Rnn()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

