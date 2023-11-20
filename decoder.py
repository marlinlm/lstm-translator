import torch.nn as nn
import torch

class Decoder(nn.Module):
    def __init__(self, device, embedding_size=300, hidden_size=900, drop_out=0.2, num_layers=4):
        super().__init__()
        self.device = device
        self.hidden_layer_size = hidden_size
        self.n_layers = num_layers
        self.embedding_size = embedding_size
        self.lstm = nn.LSTM(embedding_size, hidden_size, num_layers, batch_first=True, dropout=drop_out)
        self.softmax = nn.Softmax(dim=-1)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=0.02)
        # nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)
        # for name, param in self.linear.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)

    def forward(self, x, lengths, hidden_in, cell_in):
        
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx]
        hidden_sorted = hidden_in[:,sorted_idx]
        cell_sorted = cell_in[:,sorted_idx]

        decoder_input = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_len.cpu(), batch_first=True, enforce_sorted=True)    
        
        # x: [batch_size, x length, embeddings]
        # hidden: [n_layers, batch size, hidden size]
        # cell: [n_layers, batch size, hidden size]
        # lstm_out: [batch size, sequence size, hidden size]
        lstm_out, (hidden,cell) = self.lstm(decoder_input, (hidden_sorted, cell_sorted))
        
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        _, original_idx = sorted_idx.sort(0, descending=False)
        hidden = hidden[:,original_idx]
        cell = cell[:,original_idx]
        lstm_out = lstm_out[original_idx]
        
        return lstm_out, hidden, cell


if __name__=="__main__":
    dev = torch.device("cpu")
    a = torch.Tensor([[[1,2,3],[0,0,0]],[[4,5,6],[7,8,9]]]).to(dev)
    a_length = torch.Tensor([1,2]).long().to(dev)
    hidden_in = torch.zeros(4, 2, 6).to(dev)
    cell_in = torch.zeros(4,2,6).to(dev)
    decoder = Decoder(dev, 3, 6).to(dev)
    print(decoder(a, a_length, hidden_in, cell_in))
    
