import torch
import time
# import word2vec
import numpy as np
import corpus_reader
import tokenizer as tknzr
from seq2seq import Seq2Seq
from creterion import Creterion
import dict
import sys

max_x_len = 20
max_y_len = 20

            
def load_test_data(zh_corpus_fname, en_corpus_fname, epoch_size:int, batch_size:int, batches_per_load:int):
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

                
def train(device, model,  zh_word_2_index, zh_keys, en_word_2_index, en_keys, optimizer, zh_corpus_fname, en_corpus_fname, epoches:int, epoch_size:int, batch_size: int, out_file_prefix=""):
    loss_fun = Creterion(device)
    data_loader = load_test_data(zh_corpus_fname, en_corpus_fname, epoch_size, batch_size, 0)
    data = next(data_loader)

    loss_out_file = "./loss/" + out_file_prefix + ".loss"
    with open(loss_out_file, 'a', encoding='utf-8') as ff:
        for e in range(epoches):
            # for batch_idx, ((en,en_len),(zh, zh_len)) in enumerate(data):
            for batch_idx, (en_batch, zh_batch) in enumerate(data):
                ((en,en_len),(zh, zh_len)) = dict.scentences_to_indexes(device, en_batch, en_word_2_index), dict.scentences_to_indexes(device, zh_batch, zh_word_2_index)
                
                soft_max = model(en, en_len, zh, zh_len)
                loss = loss_fun(soft_max, zh, zh_len, batch_size)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5, norm_type=2)
                optimizer.step()
                
                loss_str = f'{time.asctime(time.localtime(time.time()))},{str(e)},{str(batch_idx)},{str(loss.item())}\n'
                ff.write(loss_str)
                if batch_idx % 100 == 0:
                    predicted = dict.softmax_to_indexes(device, soft_max, zh_idx)
                    print("\n=======================================================")
                    print(loss_str)
                    print(dict.index_to_scentence(en[0:1],en_keys))
                    print(dict.index_to_scentence(zh[0:1],zh_keys))
                    print(dict.index_to_scentence(predicted[0:1],zh_keys))
                # do_batched_train(device, model, batch, optimizer, loss, 2, 50, _b, "../loss.log")
                # do_train(device, model, batch, optimizer, loss, _b, "../loss.log")
            
            if e%10 == 0:
                model_out_name = "./models/" + out_file_prefix + "_" + str(e) + ".model"
                print(f'[{time.asctime(time.localtime(time.time()))}] - saving model:{model_out_name}')
                torch.save(model, model_out_name)
                
    model_out_name = "./models/" + out_file_prefix + "_" + str(e) + "_last" + ".model"
    print(f'[{time.asctime(time.localtime(time.time()))}] - saving model:{model_out_name}')
    torch.save(model, model_out_name)
        
if __name__=="__main__":
    device = torch.device('cuda')
    embeddings = 300
    hiddens = 600
    n_layers = 4
    out_vac_size = 20000
    zh_dict_fname = "../corpus/zh_dict.txt"
    en_dict_fname = "../corpus/en_dict.txt"
    
    print("loading dicts")
    en_idx, en_keys = dict.load_dict(en_dict_fname)
    zh_idx, zh_keys = dict.load_dict(zh_dict_fname)

    # prepare_test_data(embedding_loader, "../corpus/train", "../corpus/zh.txt", "../corpus/en.txt", 1000000)
    
    # model_fname = "./models/_seq2seq_1699753627.1307073_162"
    model_fname = None
    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')
    else:
        # model = Seq2Seq(device, embedding_loader, embeddings, hiddens, out_vac_size, n_layers, 0.5, 0.5).to(device)
        
        print("creating new model")
        model = Seq2Seq(device, en_idx, zh_idx, embeddings, hiddens, n_layers, 0.5, 0.5).to(device)
        
    optimizer = torch.optim.Adam(model.parameters())
    
    epoches = 200
    corpus = 500000
    batch_size = 100
    out_file_prefix = '20231124_' +  '_'.join([str(epoches), str(corpus), str(batch_size)])
    print("training...")
    train(device, model, zh_idx, zh_keys, en_idx, en_keys, optimizer, "../corpus/zh_paral.txt", "../corpus/en_paral.txt", 200, 500000, 100, out_file_prefix)
    