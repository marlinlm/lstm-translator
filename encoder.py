import torch.nn as nn
import torch

class Encoder(nn.Module):
    def __init__(self, device, embeddings=300, hidden_size=900, drop_out=0.2, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embeddings
        self.lstm = nn.LSTM(embeddings, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.linear = nn.Linear(hidden_size, embeddings)
        self.encoder = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.c_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.relu = nn.Tanh()

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=0.02)
        nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)
        nn.init.xavier_uniform_(self.encoder.weight.data,gain=0.25)
        # nn.init.xavier_uniform_(self.encoder_2.weight.data,gain=0.25)
        # nn.init.xavier_uniform_(self.encoder_2.weight.data,gain=0.25)

    def forward(self, x):
        # x = [batch size, seq length, embeddings]
        # x_length = [batches]
        # lstm_out = [batch size, x length, hidden size]
        # model = self.to(self.device)
        # x = x.to(self.device)
        lstm_out, (hidden, cell) = self.lstm(x)

        h = torch.cat((hidden, cell), dim=2)
        h_out = self.encoder(h)

        lineared = self.linear(lstm_out)
        out = self.relu(lineared)

        # hidden = [n_layer, batch size, hidden size]
        # cell = [n_layer, batch size, hidden size]
        # return out, lineared[:self.n_layers], lineared[self.n_layers:]
        h_out_cont = h_out[:,:,:self.hidden_layer_size].clone().detach().to(self.device)
        cell_out_cont = h_out[:,:,self.hidden_layer_size:].clone().detach().to(self.device)
        return out, h_out_cont, cell_out_cont
        # return out, hidden, cell
    



