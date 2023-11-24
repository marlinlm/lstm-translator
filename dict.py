import corpus_reader
import tokenizer as tknzr
import numpy as np
import word2vec
import torch
import re

# \xc2a0 is a special space char and need to be removed.
C2A0 = b'\xc2\xa0'.decode('utf-8')

ZH_RE = re.compile(r'^[\u4e00-\u9fa5。、：“”（）《》；！？·]+$')
EN_RE = re.compile(r'^[a-zA-Z,\.\'!\-;?/]+$')

max_x_len = 20
max_y_len = 20
max_en_vocab = 20000
max_zh_vocab = 20000
max_corpus_pair = 2000000

UNK = "<unk>"
PAD = "<pad>"
EOS = "<eos>"

def check_dict(key_2_idx:{}, keys:[]):
    for k in keys:
        if not (k in key_2_idx):
            return False
    return True
        

def add_dict(idx:{}, keys:[]):
    for k in keys:
        if k in idx:
            idx[k] += 1
        else:
            idx[k] = 1

def save_dict(idx, fname):
    with open(fname, 'a', encoding='utf-8') as f:
        for (key, val) in idx:
            # f.write(key + '\n')
            f.write(key+ " " + str(val) + '\n')


def build_freq(corpus_fname):
    tokenizer = tknzr.Tokenizer()
    reader = corpus_reader.TmxHandler()
    generator = reader.parse(corpus_fname)
    
    zh_idx = {}
    en_idx = {}
    
    length = 0
    
    try:
        while length <= max_corpus_pair:
            corpus = next(generator)
            if 'en' in corpus and 'zh' in corpus:
                en = tokenizer.tokenize(corpus['en'].lower(), lang= 'en')
                if len(en) > max_x_len:
                    continue
                zh = tokenizer.tokenize(corpus['zh'], lang='zh')
                while ' ' in zh:
                    zh.remove(' ') 
                while C2A0 in zh:
                    zh.remove(C2A0) 
                if len(zh) > max_y_len:
                    continue
                
                
                add_dict(zh_idx, zh)
                add_dict(en_idx, en)
                length += 1
                if length%10000 == 0:
                    print(str(length))
                    print("zh_dict len:", str(len(zh_idx)))
                    print("en_dict len:", str(len(en_idx)))
    except StopIteration:
        pass
    
    return zh_idx, en_idx
    

def build_dict_from_keys(keys:[]):
    dict = {}
    for i in range(len(keys)):
        k = keys[i].split()[0]
        if k in dict:
            print("duplicated key:" + k)
        dict[k] = i
    return dict

def build_dict(corpus_fname, zh_out_fname, en_out_fname):
    
    print("building freq")
    zh_freq, en_freq = build_freq(corpus_fname)

    
    print("sorting freq")
    
    zh_freq_sorted = sorted(zh_freq.items(), key = lambda x : x[1], reverse = True)
    en_freq_sorted = sorted(en_freq.items(), key = lambda x : x[1], reverse = True)

    zh_keys = zh_freq_sorted[:max_zh_vocab - 3]
    en_keys = en_freq_sorted[:max_en_vocab - 3]
    zh_keys.append((UNK,0))
    zh_keys.append((EOS,0))
    zh_keys.append((PAD,0))
    en_keys.append((UNK,0))
    en_keys.append((EOS,0))
    en_keys.append((PAD,0))
    
    print("saving dicts")
    save_dict(zh_keys, zh_out_fname)
    save_dict(en_keys, en_out_fname)
    
    return zh_keys, en_keys

def load_dict(dict_fname):
    with open(dict_fname, mode = "r", encoding='utf-8') as f:
        keys = f.readlines()
        _keys = []
        idx = build_dict_from_keys(keys)
        for k in keys:
            _keys.append(k.split()[0])
        return idx, _keys

def scentences_to_indexes(device, scents:[], w2i:{}):
    lengths = [len(scent) for scent in scents]
    max_len = max(lengths) + 1
    batches = len(lengths)
        
    idx = torch.Tensor([w2i[PAD]]).long().repeat((batches, max_len),0).to(device)
    for b, ws in enumerate(scents):
        for i, w in enumerate(ws):
            try:
                idx[b,i] = w2i[w]
            except:
                idx[b,i] = w2i[UNK]
        idx[b, lengths[b]] = w2i[EOS]
    
    lengths = torch.Tensor(lengths).long() + 1
            
    return idx.contiguous(), lengths.to(device)


def softmax_to_indexes(device, vectors, w2i:{}):
    
    batches = vectors.shape[0]
    max_len = vectors.shape[1]
        
    indexes = torch.Tensor([w2i[PAD]]).long().repeat((batches, max_len), 0).to(device)
    for b, v in enumerate(vectors):
        indexes[b,:] = torch.Tensor([w.argmax() for w in v]).to(device)
    return indexes.contiguous()

def index_to_scentence(indexes:torch.Tensor, keys:[]):
    scents = []
    
    for w in indexes:
        scent = []
        for v in w:
            try:
                scent.append(keys[v])
            except:
                scent.append(UNK)
        scents.append(scent)
    return scents
            

def prepare_corpus(corpus_fname, zh_out_fname, en_out_fname, zh_dict, en_dict, max_corpus_size=0):
    
    tokenizer = tknzr.Tokenizer()
    reader = corpus_reader.TmxHandler()
    generator = reader.parse(corpus_fname)
    
    with open(zh_out_fname, 'a', encoding='utf-8') as zh_ff:
        with open(en_out_fname, 'a', encoding='utf-8') as en_ff:
            length = 0
            try:
                while max_corpus_size == 0 or length <= max_corpus_size:
                    corpus = next(generator)
                    if 'en' in corpus and 'zh' in corpus:
                        en = tokenizer.tokenize(corpus['en'].lower(), lang= 'en')
                        if len(en) > max_x_len:
                            continue
                        zh = tokenizer.tokenize(corpus['zh'], lang='zh')
                        while ' ' in zh:
                            zh.remove(' ') 
                        while C2A0 in zh:
                            zh.remove(C2A0) 
                        if len(zh) > max_y_len:
                            continue

                        if check_dict(en_dict, en) and check_dict(zh_dict, zh):
                            zh_ff.write(' '.join(zh) + '\n')
                            en_ff.write(' '.join(en) + '\n')
                            length += 1
                            if length % 10000 == 0:
                                print(str(length))
                        
            except StopIteration:
                return

if __name__ == "__main__":
    # build_dict("../corpus/train", "../corpus/zh_dict.txt", "../corpus/en_dict.txt")

    zh_idx, zh_keys = load_dict("../corpus/zh_dict.txt")
    en_idx, en_keys = load_dict("../corpus/en_dict.txt")
    
    prepare_corpus("../corpus/train", "../corpus/zh_paral.txt_test", "../corpus/en_paral.txt_test", zh_idx, en_idx, max_corpus_size=500000)
    pass