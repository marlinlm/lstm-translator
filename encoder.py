import torch
import torch.nn as nn

# import seaborn as sns
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
# %matplotlib inline

class Encoder(nn.Module):
    def __init__(self, device, embeddings=300, hidden_size=900, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embeddings
        self.lstm = nn.LSTM(embeddings, hidden_size, num_layers, batch_first=True)
        # self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x = [batch size, seq length, embeddings]
        # lstm_out = [x length, batch size, hidden size]
        model = self.to(self.device)
        x = x.to(self.device)
        lstm_out, (hidden, cell) = model.lstm(x)


        # hidden = [n_layer, batch size, hidden size]
        # cell = [n_layer, batch size, hidden size]
        return hidden, cell


