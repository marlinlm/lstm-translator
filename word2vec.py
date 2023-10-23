from gensim.models import KeyedVectors
import torch
import jieba
import nltk
# nltk.download('punkt')
from nltk import word_tokenize
import numpy as np

class WordEmbeddingLoader():
    def __init__(self):
        pass

    def loadFromFile(self, fname):
        self.model = KeyedVectors.load_word2vec_format(fname)

    def getEmbedding(self, word:str):
        if self.model:
            try:
                return self.model.get_vector(word)
            except(KeyError):
                return None
        else:
            return None
    
    def getEmbeddingsForScentence(self, scent:str, lang:str):
        embeddings = []
        ws = []
        if(lang == 'zh'):
            ws = jieba.cut(scent, cut_all=True)
        elif lang == 'en':
            ws = word_tokenize(scent)
        else:
            raise Exception('Unsupported language ' + lang)

        for w in ws:
            embedding = self.getEmbedding(w.lower())
            if embedding is None:
                embedding = np.zeros(self.model.vector_size)

            embedding = torch.from_numpy(embedding).float()
            embeddings.append(embedding.unsqueeze(0))
        return torch.cat(embeddings, dim=0)
        
    def vector_2_word(self, vector:torch.Tensor):
        if not self.model:
            return None
        return self.model.similar_by_vector(vector.numpy())[0][0]
    
    def vector_2_scentence(self, vectors:torch.Tensor):
        if not self.model:
            return None
        scent = []
        for embedding in vectors:
            scent.append(self.vector_2_word(embedding))
        return scent
    