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
        
        if embedding_model_loader is None:
            self.embedding_loader = word2vec.WordEmbeddingLoader()
            self.embedding_loaded = False
        else:
            self.embedding_loader = embedding_model_loader
            self.embedding_loaded = True

    def load_embeddings(self, embedding_fname):
        self.embedding_loader.loadFromFile(embedding_fname)
        self.embedding_loaded = True
    
    def predict(self, sequence, length):
        x = self.embedding_loader.getEmbeddingsForScentence(sequence, 'en')

        y = torch.zeros(0, self.embeddings)
        for i in range(length):
            y_input = y.unsqueeze(0).to(self.device)
            _y = self.s2s(x,y_input)
            torch.cat(y, _y[0,i].unsqueeze())
        
        output = []
        for v_y in y:
            w_y = self.embedding_loader.search(v_y)
            output.append(w_y)

        return output
    
