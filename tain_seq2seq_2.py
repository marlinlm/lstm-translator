import torch
import time
import word2vec
import numpy as np
import corpus_reader
import tokenizer as tknzr
from seq2seq import Seq2Seq
from creterion import Creterion
import sys

max_x_len = 20
max_y_len = 20

def prepare_test_data(embedding_loader:word2vec.WordEmbeddingLoader, corpus_fname, zh_out_fname, en_out_fname, max_len=0):
    
    tokenizer = tknzr.Tokenizer()
    reader = corpus_reader.TmxHandler()
    generator = reader.parse(corpus_fname)
    
    with open(zh_out_fname, 'a', encoding='utf-8') as zh_ff:
        with open(en_out_fname, 'a', encoding='utf-8') as en_ff:
            length = 0
            try:
                while max_len == 0 or length <= max_len:
                    corpus = next(generator)
                    if 'en' in corpus and 'zh' in corpus:
                        en = tokenizer.tokenize(corpus['en'].lower(), lang= 'en')
                        if len(en) > max_x_len:
                            continue
                        zh = tokenizer.tokenize(corpus['zh'], lang='zh')
                        if len(zh) > max_y_len:
                            continue
                        if embedding_loader.scentence_vocab_check(en, 'en') and embedding_loader.scentence_vocab_check(zh, 'zh'):
                            zh_ff.write(' '.join(zh) + '\n')
                            en_ff.write(' '.join(en) + '\n')
                            length += 1
            except StopIteration:
                return
            
def load_test_data(embedding_loader:word2vec.WordEmbeddingLoader, zh_corpus_fname, en_corpus_fname, epoch_size:int, batch_size:int, batches_per_load:int):
    reader = corpus_reader.CorpusReader()
    data = []
    end = False
    generator = reader.parse(zh_corpus_fname, en_corpus_fname)
    for _ in np.arange(0, epoch_size, batch_size):
        en_batch = []
        zh_batch = []
        length = 0
        while length < batch_size:
            try:
                (zh,en) = next(generator)
                en_batch.append(en)
                zh_batch.append(zh)
                length += 1
            except StopIteration:
                end = True
                break
        
        data.append((en_batch, zh_batch))
        if (batches_per_load > 0) and (len(data) % batches_per_load == 0):
            yield data
            data = []
        if end:
            break
    
    yield data

def load_test_data_from_tmx(embedding_loader:word2vec.WordEmbeddingLoader, corpus_fname, epoch_size:int, batch_size:int, batches_per_load:int):

    tokenizer = tknzr.Tokenizer()
    reader = corpus_reader.TmxHandler()
    
    data = []
    end = False
    generator = reader.parse(corpus_fname)
    for _ in np.arange(0, epoch_size, batch_size):
        en_batch = []
        zh_batch = []
        length = 0
        while length < batch_size:
            try:
                corpus = next(generator)
                if 'en' in corpus and 'zh' in corpus:
                    en = tokenizer.tokenize(corpus['en'].lower(), lang= 'en')
                    if len(en) > max_x_len:
                        continue
                    zh = tokenizer.tokenize(corpus['zh'], lang='zh')
                    if len(zh) > max_y_len:
                        continue
                    if embedding_loader.scentence_vocab_check(en, 'en') and embedding_loader.scentence_vocab_check(zh, 'zh'):
                        en_batch.append(en)
                        zh_batch.append(zh)
                        length += 1
            except StopIteration:
                end = True
                break
        
        data.append((en_batch, zh_batch))
        if (batches_per_load > 0) and (len(data) % batches_per_load == 0):
            yield data
            data = []
        if end:
            break
    
    yield data
                
def train(device, model, embedding_loader:word2vec.WordEmbeddingLoader, optimizer, zh_corpus_fname, en_corpus_fname, epoches:int, epoch_size:int, batch_size: int, loss_out_file="../loss.log"):
    loss_fun = Creterion(device)
    data_loader = load_test_data(embedding_loader, zh_corpus_fname, en_corpus_fname, epoch_size, batch_size, 0)
    data = next(data_loader)
    with open(loss_out_file, 'a', encoding='utf-8') as ff:
        for e in range(epoches):
            # for batch_idx, ((en,en_len),(zh, zh_len)) in enumerate(data):
            for batch_idx, (en_batch, zh_batch) in enumerate(data):
                ((en,en_len, en_know),(zh, zh_len, zh_know)) = embedding_loader.scentences_to_indexes(en_batch, 'en'), embedding_loader.scentences_to_indexes(zh_batch, 'zh')
                
                en_emb = embedding_loader.get_embeddings(en, lang="en")
                zh_emb = embedding_loader.get_embeddings(zh, lang="zh")
                soft_max = model(en_emb, en_len - 1, zh_emb, zh_len - 1)
                loss = loss_fun(soft_max, zh[:,:-1], zh_len - 1)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                
                loss_str = f'{time.asctime(time.localtime(time.time()))},{str(e)},{str(batch_idx)},{str(loss.item())}\n'
                ff.write(loss_str)
                print(loss_str)
                # do_batched_train(device, model, batch, optimizer, loss, 2, 50, _b, "../loss.log")
                # do_train(device, model, batch, optimizer, loss, _b, "../loss.log")
            
            if e%10 == 0:
                model_out_name = "./models/seq2seq_" + str(time.time()) + "_" + str(e)
                print(f'[{time.asctime(time.localtime(time.time()))}] - saving model:{model_out_name}')
                torch.save(model, model_out_name)
                
    model_out_name = "./models/seq2seq_" + str(time.time()) + "_" + "last"
    print(f'[{time.asctime(time.localtime(time.time()))}] - saving model:{model_out_name}')
    torch.save(model, model_out_name)
        
if __name__=="__main__":
    device = torch.device('cuda')
    embeddings = 300
    hiddens = 600
    n_layers = 4
    out_vac_size = 20000

    print("loading embedding")
    embedding_loader = word2vec.WordEmbeddingLoader(device, "../embeddings/sgns.merge/sgns.merge.word", out_vocab_size=out_vac_size)
    # embedding_loader = word2vec.WordEmbeddingLoader("../embeddings/parellel_01.v2c")
    print("load embedding finished")
    
    # prepare_test_data(embedding_loader, "../corpus/train", "../corpus/zh.txt", "../corpus/en.txt", 1000000)
    
    # model_fname = "./models/_seq2seq_1699753627.1307073_162"
    model_fname = None
    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')
    else:
        model = Seq2Seq(device, embedding_loader, embeddings, hiddens, out_vac_size, n_layers, 0.5, 0.5).to(device)
        
    optimizer = torch.optim.Adam(model.parameters())
    
    train(device, model, embedding_loader, optimizer, "../corpus/zh.txt", "../corpus/en.txt", 200, 500000, 100)
    