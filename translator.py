import seq2seq
import word2vec
import encoder
import decoder
import corpus_reader
import time
import numpy as np
import torch
import torch.nn


class Translator():
    def __init__(self, dev, embeddings, hiddens, n_layers, model, embedding_model_loader):
        super().__init__()
        self.device = dev
        # self.reader = corpus_reader.tmxHandler()
        self.embeddings = embeddings
        if model is None:
            self.s2s = seq2seq.Seq2Seq(dev, embeddings, hiddens, n_layers)
        else:
            self.s2s = model
        
        if embedding_model_loader is None:
            self.embedding_loader = word2vec.WordEmbeddingLoader()
        else:
            self.embedding_loader = embedding_model_loader
        self.embedding_loaded = False

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

    def do_train(self, corpus_fname, batch_size:int, batches: int):
        reader = corpus_reader.tmxHandler()
        loss = torch.nn.MSELoss()
        model = self.s2s.to(self.device)
        print(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        generator = reader.parse(corpus_fname)
        for _b in range(batches):
            batch = []
            try:
                for _c in range(batch_size):
                    try:
                        corpus = next(generator)
                        if 'en' in corpus and 'zh' in corpus:
                            en = self.embedding_loader.getEmbeddingsForScentence(corpus['en'],'en')
                            zh = self.embedding_loader.getEmbeddingsForScentence(corpus['zh'], 'zh')
                            batch.append((en,zh))
                    except (StopIteration):
                        break
            finally:
                print(time.localtime())
                print("batch: " + str(_b))
                model.do_train(batch, optimizer, loss)

if __name__=="__main__":
    device = torch.device('cuda')
    embeddings = 300
    hiddens = 600
    n_layers = 4
    # model_fname = "./models/model_latest"
    model_fname = None

    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')

    translator = Translator(device, embeddings, hiddens, n_layers, model)
    print('loading embeddings')
    translator.load_embeddings("../sgns.merge.word")
    print('training')
    translator.do_train("../News-Commentary_v16.tmx", 1000, 100)

