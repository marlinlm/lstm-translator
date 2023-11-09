import torch
import encoder as enc
import decoder as dec
import torch.nn as nn

class Seq2Seq(nn.Module):
    def __init__(self, device, embeddings, hiddens, n_layers, encoder_dropout, decoder_dropout):
        super().__init__()
        self.device = device
        self.encoder = enc.Encoder(device, embeddings, hiddens, encoder_dropout, n_layers)
        self.decoder= dec.Decoder(device, embeddings, hiddens, decoder_dropout, n_layers)
        self.embeddings = self.encoder.embedding_size
        assert self.encoder.n_layers == self.decoder.n_layers, "Number of layers of encoder and decoder must be equal!"
        assert self.decoder.hidden_layer_size==self.decoder.hidden_layer_size, "Hidden layer size of encoder and decoder must be equal!"

    # x: [batches, x length, embeddings]
    # y: [batches, y length, embeddings]
    def forward(self, x, y):

        # encoder_out: [batches, n_layers, embeddings]
        # hidden, cell: [n layers, batch size, embeddings]
        encoder_out, hidden, cell = self.encoder(x)

        # use encoder output as the first word of the decode sequence
        decoder_input = torch.cat((encoder_out[:, -1, :].unsqueeze(1), y), dim=1)

        # predicted: [batches, y length + 1, embeddings]
        predicted, hidden, cell = self.decoder(decoder_input, hidden, cell)

        return predicted
