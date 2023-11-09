import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, device, embedding_size=300, hidden_size=900, drop_out=0.2, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.linear = nn.Linear(hidden_size, embedding_size)
        # self.linear_2 = nn.Linear(10000, embedding_size)
        self.relu = nn.Tanh()

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=0.02)
        nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)

    def forward(self, x, hidden_in, cell_in):

        # x: [batch_size, x length, embeddings]
        # hidden: [n_layers, batch size, hidden size]
        # cell: [n_layers, batch size, hidden size]
        # lstm_out: [seq length, batch size, hidden size]
        lstm_out, (hidden,cell) = self.lstm(x, (hidden_in, cell_in))

        # prediction: [seq length, batch size, embeddings]
        prediction=self.relu(self.linear(lstm_out))
        return prediction, hidden, cell
