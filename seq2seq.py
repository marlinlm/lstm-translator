import torch
import encoder as enc
import decoder as dec
import torch.nn as nn
import word2vec

class Seq2Seq(nn.Module):
    def __init__(self, device, embedding_loader:word2vec.WordEmbeddingLoader, embeddings, hiddens, out_vocab_size, n_layers, encoder_dropout, decoder_dropout):
        super().__init__()
        self.device = device
        self.w2v = embedding_loader
        self.encoder = enc.Encoder(device, embeddings, hiddens, encoder_dropout, n_layers)
        self.decoder= dec.Decoder(device, embeddings, hiddens, decoder_dropout, n_layers)
        self.linear = nn.Linear(hiddens, out_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        self.embeddings = self.encoder.embedding_size
   
        assert self.encoder.n_layers == self.decoder.n_layers, "Number of layers of encoder and decoder must be equal!"
        assert self.decoder.hidden_layer_size==self.decoder.hidden_layer_size, "Hidden layer size of encoder and decoder must be equal!"

    # x: [batches, x length, embeddings]
    # y: [batches, y length, embeddings]
    def forward(self, x, x_length, y, y_length):
        
        assert x.shape[0] == y.shape[0], "dim 0 of x and y should be the same."
        assert x_length.shape[0] == y_length.shape[0], "dim 0 of x_length and y_length must be the same"

        # encoder_out: [batch size, seq length, embeddings]
        # hidden, cell: [n layers, batch size, hiddens]
        encoder_out, hidden, cell = self.encoder(x, x_length)
        
        encoder_lineared = self.linear(encoder_out)
        encoder_softmax = self.softmax(encoder_lineared)
        encoder_idx = self.w2v.softmax_to_indexes(encoder_softmax)
        encoder_emb = self.w2v.get_embeddings(encoder_idx, lang='zh')
        
        decoder_input = torch.cat((encoder_emb, y[:,:-1]), dim=1)
        decoder_out, _, _ = self.decoder(decoder_input, y_length, hidden, cell)
        
        softmax_out = self.linear(decoder_out)
        softmax_out = self.softmax(softmax_out)
        return softmax_out
    
    def translate(self, x, x_length, max_y_length):
        
        
        # encoder_out: [batch size, seq length, embeddings]
        # hidden, cell: [n layers, batch size, hiddens]
        encoder_out, hidden, cell = self.encoder(x, x_length)
        
        encoder_lineared = self.linear(encoder_out)
        encoder_softmax = self.softmax(encoder_lineared)
        encoder_idx = self.w2v.softmax_to_indexes(encoder_softmax)
        encoder_emb = self.w2v.get_embeddings(encoder_idx, lang='zh')
        
        y_length = torch.ones((x.shape[0])).to(self.device)
        decoder_input = encoder_emb
        output = []
        
        for _ in range(max_y_length):
            
            decoder_out, hidden, cell = self.decoder(decoder_input, y_length, hidden, cell)
            
            softmax_out = self.linear(decoder_out)
            softmax_out = self.softmax(softmax_out)
            output.append(softmax_out)
            
            s2s_out_idx = self.w2v.softmax_to_indexes(softmax_out)
            decoder_input = self.w2v.get_embeddings(s2s_out_idx, lang='zh')
        
        return torch.cat(output, dim=1)
        

if __name__=="__main__":
    dev = torch.device("cpu")
    embedding_loader = word2vec.WordEmbeddingLoader(dev, "../embeddings/sgns.merge/sgns.merge.word.toy", embeddings = 3)
    # embedding_loader = word2vec.DummyWordEmbeddingLoader(dev, 3, 0, 20)
    s2s = Seq2Seq(dev, embedding_loader, 3, 6, 20, 4, 0.5, 0.5).to(dev)
    
    x = torch.Tensor([[[1,2,3],[4,5,6]],[[21,22,23],[0,0,0]]]).to(dev)
    x_len = torch.Tensor([2,1]).long().to(dev)
    y = torch.Tensor([[[7,8,9],[10,11,12],[14,15,16]],[[27,28,29],[0,0,0],[0,0,0]]]).to(dev)
    y_len = torch.Tensor([3,1]).long().to(dev)
    # decoded = s2s(x, x_len, y, y_len)
    # print(decoded)
    translated = s2s.translate(x, x_len, 10)
    print(translated)
