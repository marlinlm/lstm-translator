from gensim.models import Word2Vec
import torch
from tokenizer import EOS_EN
from tokenizer import EOS_ZH
from tokenizer import Tokenizer
import numpy as np

class WordEmbeddingLoader():
    def __init__(self, model_fname, embeddings=300, is_word2vec_format=True, is_word2vec_binary=False):
        self.tokenizer = Tokenizer()
        if(is_word2vec_format):
            self.model = Word2Vec(vector_size=embeddings)
            self.model = self.model.wv.load_word2vec_format(model_fname)
        else:
            self.model = Word2Vec.load(model_fname).wv

    def get_embeddings(self, word:str):
        if self.model:
            try:
                return self.model.get_vector(word)
            except(KeyError):
                return None
        else:
            return None
    
    def get_scentence_embeddings(self, scent:str, lang:str):
        embeddings = []
        ws = self.tokenizer.tokenize(scent, lang)
        for w in ws:
            embedding = self.get_embeddings(w.lower())
            if embedding is None:
                embedding = np.zeros(self.model.vector_size)

            embedding = torch.from_numpy(embedding).float()
            embeddings.append(embedding.unsqueeze(0))
        return torch.cat(embeddings, dim=0)
        
    def vector_2_word(self, vector:torch.Tensor):
        if not self.model:
            return None
        return self.model.wv.similar_by_vector(vector.numpy())[0][0]
    
    def vector_2_scentence(self, vectors:torch.Tensor):
        if not self.model:
            return None
        scent = []
        for embedding in vectors:
            scent.append(self.vector_2_word(embedding))
        return scent

    

if __name__ == '__main__':
    w2v_eos = WordEmbeddingLoader("../sgns.merge.word.eos")
    print(w2v_eos.get_embeddings(EOS_ZH))
    print(w2v_eos.get_embeddings(EOS_EN))
    print(w2v_eos.get_embeddings('是'))