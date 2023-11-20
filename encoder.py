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
        self.encoder = nn.Linear(hidden_size * 2, hidden_size * 2)
        # self.c_linear = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.active = nn.Tanh()

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_uniform_(param,gain=0.02)
        # nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)
        # for name, param in self.linear.named_parameters():
        #     if 'bias' in name:
        #         nn.init.constant_(param, 0.0)
        nn.init.xavier_uniform_(self.encoder.weight.data,gain=0.25)
        for name, param in self.encoder.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)

    # x = [batch size, seq length, embeddings]
    # x_length = [batches]
    def forward(self, x, lengths):
        
        batches = x.shape[0]
        
        sorted_len, sorted_idx = lengths.sort(0, descending=True)
        x_sorted = x[sorted_idx.long()]
        
        encoder_in = nn.utils.rnn.pack_padded_sequence(x_sorted, sorted_len.cpu(), batch_first=True, enforce_sorted=True)    
        
        # lstm_out = [batch size, x length, hidden size]
        # hidden = [n_layer, batch size, hidden size]
        # cell = [n_layer, batch size, hidden size]
        lstm_out, (hidden, cell) = self.lstm(encoder_in)
        
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
                
        _, original_idx = sorted_idx.sort(0, descending=False)

        hidden = hidden[:,original_idx]
        cell = cell[:,original_idx]
        # h = torch.cat((hidden, cell), dim=2)
        # h_out = self.encoder(h)
        # h_out_cont = h_out[:,:,:self.hidden_layer_size].clone().detach().to(self.device)
        # cell_out_cont = h_out[:,:,self.hidden_layer_size:].clone().detach().to(self.device)

        lstm_out = lstm_out[original_idx]
        encoder_out = torch.zeros(size=(batches, 1, lstm_out.shape[-1])).to(lstm_out.device)
        for idx, v in enumerate(lstm_out):
            encoder_out[idx][0] = v[lengths[idx] - 1]
        
        # encoder_out = self.linear(encoder_out)
        # encoder_out = self.active(encoder_out)

        # return encoder_out, h_out_cont, cell_out_cont
        return encoder_out, hidden.contiguous(), cell.contiguous()

        # return out, hidden, cell

if __name__=="__main__":
    dev = torch.device("cpu")
    a = torch.Tensor([[[1,2,3],[0,0,0]],[[4,5,6],[7,8,9]]]).to(dev)
    a_length = torch.Tensor([1,2]).long().to(dev)
    encoder = Encoder(dev, 3, 6).to(dev)
    print(encoder(a, a_length))
    



