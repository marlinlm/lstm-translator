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
        # for i in range(length):
        #     x_input = x.unsqueeze(0).to(self.device)
        #     y_input = y.unsqueeze(0).to(self.device)
        #     _y = self.s2s(x_input,y_input)
            # torch.cat(y, _y[0,i].unsqueeze())
        x_input = x.unsqueeze(0).to(self.device)
        y_input = y.unsqueeze(0).to(self.device)
        _y = self.s2s(x_input,y_input)
        
        # output = []
        # for v_y in _y[0]:
        #     w_y = self.embedding_loader.vector_2_scentence(v_y)
        #     output.append(w_y)
        output = self.embedding_loader.vector_2_scentence(_y[0])
        return output



if __name__=='__main__':
    embedding_loader = word2vec.WordEmbeddingLoader("../embeddings/sgns.merge/sgns.merge.word")
    dev = torch.device('cuda')
    embeddings = 300
    hiddens = 1200
    n_layers = 4
    model_fname = "../models/_seq2seq_1699630089.3966775"
    model = torch.load(model_fname)
    trn = Translator(dev, embeddings, hiddens, n_layers, model, embedding_loader)
    print(trn.predict("i love China.",10))