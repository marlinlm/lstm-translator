from gensim.models import Word2Vec
import torch
import re

ZH_RE = re.compile(r'^[\u4e00-\u9fa5。、：“”（）《》；！？·]+$')
EN_RE = re.compile(r'^[a-zA-Z,\.\'!\-;?/]+$')
class WordEmbeddingLoader():
    def __init__(self, device, model_fname, embeddings=300, in_vocab_size = 0, out_vocab_size = 10000, is_word2vec_format=True, is_word2vec_binary=False):
        self.device = device
        self.embeddings = embeddings
        # self.tokenizer = Tokenizer()
        
        self.EOS = '<eos>'
        self.UNK = '<unk>'
        self.PAD = '<pad>'
        
        self.EOS_EMB = torch.zeros(embeddings).fill_(2).unsqueeze(0).to(device)
        self.UNK_EMB = torch.ones(embeddings).unsqueeze(0).to(device)
        self.PAD_EMB = torch.zeros(embeddings).unsqueeze(0).to(device)
        
        if(is_word2vec_format):
            self.model = Word2Vec(vector_size=embeddings)
            self.model = self.model.wv.load_word2vec_format(model_fname)
        else:
            self.model = Word2Vec.load(model_fname).wv
        
        self.zh_keys = []
        self.en_keys = []
        self.zh_w2i = {}
        self.en_w2i = {}
        zh_idx = 0
        en_idx = 0
        self.zh_vectors = []
        self.en_vectors = []
        for k in self.model.index_to_key:
            
            try:
                if ZH_RE.match(k):
                    
                    if out_vocab_size != 0 and zh_idx >= out_vocab_size - 3:
                        continue
                    
                    self.zh_keys.append(k)
                    self.zh_vectors.append(torch.from_numpy(self.model.get_vector(k))[:embeddings].unsqueeze(0).to(device))
                    self.zh_w2i[k]=torch.LongTensor((zh_idx,)).to(device)
                    zh_idx += 1
                elif EN_RE.match(k):
                    
                    if in_vocab_size != 0 and en_idx >= in_vocab_size - 3:
                        continue
                    
                    self.en_keys.append(k)
                    self.en_vectors.append(torch.from_numpy(self.model.get_vector(k))[:embeddings].unsqueeze(0).to(device))
                    self.en_w2i[k]=torch.LongTensor((en_idx,)).to(device)
                    en_idx += 1
                else:
                    continue
            except:
                continue
        
        self.zh_w2i[self.EOS] = torch.LongTensor((len(self.zh_keys),)).to(device)
        self.zh_keys.append(self.EOS)
        self.zh_vectors.append(self.EOS_EMB)

        self.zh_w2i[self.PAD] = torch.LongTensor((len(self.zh_keys),)).to(device)
        self.zh_keys.append(self.PAD)
        self.zh_vectors.append(self.PAD_EMB)
        
        self.zh_w2i[self.UNK] = torch.LongTensor((len(self.zh_keys),)).to(device)
        self.zh_keys.append(self.UNK)
        self.zh_vectors.append(self.UNK_EMB)

        self.embedding_zh = torch.nn.Embedding(num_embeddings=len(self.zh_keys), embedding_dim=embeddings)
        self.embedding_zh.weight.data.copy_(torch.cat(self.zh_vectors, dim=0))
        self.embedding_zh.to(device)
        self.embedding_zh.weight.requires_grad = False
        
        
        self.en_w2i[self.EOS] = torch.LongTensor((len(self.en_keys),)).to(device)
        self.en_keys.append(self.EOS)
        self.en_vectors.append(self.EOS_EMB)
        
        self.en_w2i[self.PAD] = torch.LongTensor((len(self.en_keys),)).to(device)
        self.en_keys.append(self.PAD)
        self.en_vectors.append(self.PAD_EMB)
        
        self.en_w2i[self.UNK] = torch.LongTensor((len(self.en_keys),)).to(device)
        self.en_keys.append(self.UNK)
        self.en_vectors.append(self.UNK_EMB)
        
        self.embedding_en = torch.nn.Embedding(num_embeddings=len(self.en_keys), embedding_dim=embeddings)
        self.embedding_en.weight.data.copy_(torch.cat(self.en_vectors, dim=0))
        self.embedding_en.to(device)
        self.embedding_en.weight.requires_grad = False
    
    def scentence_vocab_check(self, scent:[], lang):
        if lang == 'zh':
            w2i = self.zh_w2i
        elif lang == 'en':
            w2i = self.en_w2i
        for i, w in enumerate(scent):
            try:
                idx = w2i[w]
            except:
                return False
        return True
    
    def scentences_to_indexes(self, scents:[], lang='en'):
        if lang == 'zh':
            w2i = self.zh_w2i
        elif lang == 'en':
            w2i = self.en_w2i
        
        lengths = [len(scent) for scent in scents]
        max_len = max(lengths) + 1
        batches = len(lengths)
        all_known_words = []
            
        idx = w2i[self.PAD].repeat((batches, max_len),0).to(self.device)
        for b, ws in enumerate(scents):
            contains_unk = False
            for i, w in enumerate(ws):
                try:
                    idx[b,i] = w2i[w]
                except:
                    contains_unk = True
                    idx[b,i] = w2i[self.UNK]
            idx[b, lengths[b]] = w2i[self.EOS]
            if not contains_unk:
                all_known_words.append(b)        
        
        lengths = torch.Tensor(lengths).long() + 1
                
        return idx.contiguous(), lengths.to(self.device), all_known_words
    
    def get_embeddings(self, idexes, lang = "en"):
        if lang == 'zh':
            embedding = self.embedding_zh
        elif lang == 'en':
            embedding = self.embedding_en
            
        return embedding(idexes)[:,:,:self.embeddings]
    
    def softmax_to_indexes(self, vectors, lang = "zh"):
        if lang == "en":
            w2i = self.en_w2i
        elif lang == "zh":
            w2i = self.zh_w2i
        
        batches = vectors.shape[0]
        max_len = vectors.shape[1]
            
        indexes = w2i[self.PAD].repeat((batches, max_len), 0).to(self.device)
        for b, v in enumerate(vectors):
            indexes[b,:] = torch.Tensor([w.argmax() for w in v]).to(self.device)
        return indexes.contiguous()
    
    def index_to_scentence(self, indexes:torch.Tensor, lang = "zh"):
        scents = []
        
        if lang == 'en':
            keys = self.en_keys
        elif lang == 'zh':
            keys = self.zh_keys
        
        for w in indexes:
            scent = []
            for v in w:
                try:
                    scent.append(keys[v])
                except:
                    scent.append(self.UNK)
            scents.append(scent)
        return scents
                
class DummyWordEmbeddingLoader(WordEmbeddingLoader):
    def __init__(self, device, embeddings=300, in_vocab_size = 0, out_vocab_size = 10000):
        self.device = device
        self.embeddings = embeddings
        self.in_vocab_size = 0
        self.out_vocab_size = 0
    
    def scentences_to_indexes(self, ws, lang='en'):
        lenths = [len(w) for w in ws]
        idx = torch.arange(0, max(lenths) * len(ws)).view(len(ws), -1).to(self.device)
        return idx
    
    def get_embeddings(self, idexes, lang = "en"):
        return torch.randn((idexes.shape[0], idexes.shape[1], self.embeddings)).to(self.device)
    
    def softmax_to_indexes(self, vectors, lang = "zh"):
        return torch.arange(0, vectors.shape[0] * vectors.shape[1]).view(vectors.shape[0], vectors.shape[1]).to(self.device)
    
    def index_to_scentence(self, indexes:torch.Tensor, lang = "zh"):
        scent = []
        for i in range(len(indexes)):
            scent.append(['banana' for _ in indexes[i]])        
        return scent
                

if __name__ == '__main__':
    dev = torch.device("cuda")
    # w2v = WordEmbeddingLoader(dev, "../embeddings/sgns.merge/sgns.merge.word.toy")
    w2v = WordEmbeddingLoader(dev, "../embeddings/sgns.merge/sgns.merge.word.toy", 3)
    # print(w2v_eos.get_embeddings(EOS_ZH))
    # print(w2v_eos.get_embeddings(EOS_EN))
    # print(w2v_eos.get_embeddings('是'))
    indexes = w2v.scentences_to_indexes(['我喜欢踢足球。','好。。。。'], 'zh')
    print(indexes)
    print(w2v.index_to_scentence(indexes, 'zh'))
    print(w2v.get_embeddings(indexes, 'zh'))
    softmax = torch.Tensor([[[1,2,3],[4,6,5]],[[7,8,9],[0,0,0]]]).to(dev)
    indexes = w2v.softmax_to_indexes(softmax)
    print(indexes)
    print(w2v.index_to_scentence(indexes, 'zh'))