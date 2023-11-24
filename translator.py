import seq2seq
import word2vec
import encoder
import decoder
import corpus_reader
import time
import numpy as np
import torch
import torch.nn
import time
from seq2seq import Seq2Seq

from torchinfo import summary


class Translator():
    def __init__(self, dev, embeddings, hiddens, n_layers, model, embedding_model_loader):
        super().__init__()
        self.device = dev
        # self.reader = corpus_reader.tmxHandler()
        self.embeddings = embeddings
        if model is None:
            self.s2s = seq2seq.Seq2Seq(dev, embeddings, hiddens, n_layers).to(dev)
        else:
            self.s2s = model
        
        # if embedding_model_loader is None:
        #     self.embedding_loader = word2vec.WordEmbeddingLoader()
        #     self.embedding_loaded = False
        # else:
        self.embedding_loader = embedding_model_loader
        self.embedding_loaded = True

    
    def predict(self, sequence, length):
        x = self.embedding_loader.get_scentence_embeddings(sequence, 'en')
        y = torch.zeros(length, self.embeddings)
        y_input = torch.zeros(1, length, self.s2s.embeddings)
        
        x_input = torch.zeros(1, length, self.s2s.embeddings)
        x_input[0, :x.shape[0]] = x
        
        for i in range(length):
            _y = self.s2s(x_input.to(self.device), y_input.to(self.device))
            y_input[0, i] = _y[0, i]
            print(self.embedding_loader.vector_2_scentence(y_input[0]))
            print(self.embedding_loader.vector_2_scentence(_y[0]))

        # output = []
        # for v_y in _y[0]:
        #     w_y = self.embedding_loader.vector_2_scentence(v_y)
        #     output.append(w_y)
        output = self.embedding_loader.vector_2_scentence(y_input[0])
        return output



if __name__=='__main__':
    out_vac_size = 20000
    device = torch.device('cuda')
    print("loading embedding")
    embedding_loader = word2vec.WordEmbeddingLoader(device, "../embeddings/sgns.merge/sgns.merge.word", out_vocab_size=out_vac_size)
    print("loaded embedding")
    
    embeddings = 300
    hiddens = 600
    n_layers = 4

    model_fname = "../models/_seq2seq_1700566502.6062691_120"
    # model_fname = None
    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')
    else:
        model = Seq2Seq(device, embedding_loader, embeddings, hiddens, out_vac_size, n_layers, 0.5, 0.5).to(device)
    
    model.eval()

    import tokenizer as tknzr
    tokenizer = tknzr.Tokenizer()

    x = []
    # x.append(tokenizer.tokenize("public opinion will always focus on our failures.", lang= 'en'))
    # x.append(tokenizer.tokenize("she would like the wording of the resolutions to be checked.", lang= 'en'))
    # x.append(tokenizer.tokenize("information technology strategy costs.", lang= 'en'))
    # x.append(tokenizer.tokenize("these principles were not fully implemented.", lang= 'en'))
    # x.append(tokenizer.tokenize("a number of payment transactions were made without prior approval.", lang='en'))
    x.append(tokenizer.tokenize("number", lang='en'))

    # x.append(tokenizer.tokenize("medical", lang= 'en'))
    print(x)
    x_idx, x_len, _ = embedding_loader.scentences_to_indexes(x, lang="en")
    print("x_idx:", x_idx)
    print("x_len:", x_len)
    x_emb = embedding_loader.get_embeddings(x_idx, lang="en")
    # print("x_emb:", x_emb)
    while True:
        y_softmax = model.translate(x_emb, x_len, 20)
        print("y_softmax:", y_softmax)
        y_idx = embedding_loader.softmax_to_indexes(y_softmax, lang="zh")
        translated = embedding_loader.index_to_scentence(y_idx, lang="zh")
        print(translated)