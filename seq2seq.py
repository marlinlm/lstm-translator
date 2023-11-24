import torch
import encoder as enc
import decoder as dec
import torch.nn as nn
# import word2vec
import dict

class Seq2Seq(nn.Module):
    def __init__(self, device, en_word_2_idx, zh_word_2_idx, embeddings, hiddens, n_layers, encoder_dropout, decoder_dropout):
        super().__init__()
        self.device = device
        # self.w2v = embedding_loader
        self.src_idx = en_word_2_idx
        self.tar_idx = zh_word_2_idx
        src_vocab_size = len(self.src_idx)
        tar_vocab_size = len(self.tar_idx)
        self.encoder = enc.Encoder(device, src_vocab_size, embeddings, hiddens, encoder_dropout, n_layers)
        self.decoder= dec.Decoder(device, tar_vocab_size, embeddings, hiddens, decoder_dropout, n_layers)
        self.linear = nn.Linear(hiddens, tar_vocab_size)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.xavier_uniform_(self.linear.weight.data,gain=0.25)
        for name, param in self.linear.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
                
        self.embeddings = self.encoder.embedding_size
   
        assert self.encoder.n_layers == self.decoder.n_layers, "Number of layers of encoder and decoder must be equal!"
        assert self.decoder.hidden_layer_size==self.decoder.hidden_layer_size, "Hidden layer size of encoder and decoder must be equal!"

    # x: [batches, x length, 1]
    # y: [batches, y length, 1]
    def forward(self, x, x_length, y, y_length):
        
        assert x.shape[0] == y.shape[0], "dim 0 of x and y should be the same."
        assert x_length.shape[0] == y_length.shape[0], "dim 0 of x_length and y_length must be the same"

        # encoder_out: [batch size, seq length, embeddings]
        # hidden, cell: [n layers, batch size, hiddens]
        encoder_out, hidden, cell = self.encoder(x, x_length)
        
        encoder_lineared = self.linear(encoder_out)
        encoder_softmax = self.softmax(encoder_lineared)
        encoder_idx = dict.softmax_to_indexes(self.device, encoder_softmax, self.tar_idx)
        
        decoder_input = torch.cat((encoder_idx, y[:,:-1]), dim=1)
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
        encoder_idx = dict.softmax_to_indexes(self.device, encoder_softmax, self.tar_idx)
        
        y_length = torch.ones((x.shape[0])).to(self.device)
        decoder_input = encoder_idx
        output = []
        
        for _ in range(max_y_length):
            
            decoder_out, hidden, cell = self.decoder(decoder_input, y_length, hidden, cell)
            
            softmax_out = self.linear(decoder_out)
            softmax_out = self.softmax(softmax_out)
            output.append(softmax_out)
            
            s2s_out_idx = dict.softmax_to_indexes(self.device, softmax_out, self.tar_idx)
            decoder_input = s2s_out_idx
        
        return torch.cat(output, dim=1)
        

if __name__=="__main__":
    src_vocab_fname = "../corpus/en_dict.txt"
    tar_vocab_fname = "../corpus/zh_dict.txt"
    dev = torch.device("cpu")
    
    src_idx, src_keys = dict.load_dict(src_vocab_fname)
    tar_idx, tar_keys = dict.load_dict(tar_vocab_fname)
    
    # embedding_loader = word2vec.WordEmbeddingLoader(dev, "../embeddings/sgns.merge/sgns.merge.word.toy", embeddings = 3)
    # embedding_loader = word2vec.DummyWordEmbeddingLoader(dev, 3, 0, 20)
    s2s = Seq2Seq(dev, src_idx, tar_idx, 3, 6, 4, 0.5, 0.5).to(dev)
    
    x = torch.Tensor([[1,2],[3,0]]).long().to(dev)
    x_len = torch.Tensor([2,1]).long().to(dev)
    y = torch.Tensor([[12,13,14],[5,0,0]]).long().to(dev)
    y_len = torch.Tensor([3,1]).long().to(dev)
    # decoded = s2s(x, x_len, y, y_len)
    # print(decoded)
    translated = s2s.translate(x, x_len, 10)
    print(translated)
