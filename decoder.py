import torch
import torch.nn as nn

# import seaborn as sns
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
# %matplotlib inline

class Decoder(nn.Module):
    def __init__(self, device, embedding_size=300, hidden_size=900, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, embedding_size)

    def forward(self, x, hidden_in, cell_in):

        model = self.to(self.device)
        x = x.to(self.device)
        hidden_in = hidden_in.to(self.device)
        cell_in = cell_in.to(self.device)

        # x = [batch_size, x length, embeddings]
        # hidden = [n_layers, batch size, hidden size]
        # cell = [n_layers, batch size, hidden size]
        # lstm_out = [seq length, batch size, hidden size]
        lstm_out, (hidden,cell) = model.lstm(x, (hidden_in, cell_in))

        # prediction = [seq length, batch size, embeddings]
        prediction=model.linear(lstm_out)
        return prediction, hidden, cell
