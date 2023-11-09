import torch
import time
import word2vec
import numpy as np
import corpus_reader
import tokenizer
from seq2seq import Seq2Seq
import sys

max_x_len = 200
max_y_len = 200

def do_train(device, model:Seq2Seq, train_set, optimizer, loss_function, batch_idx, loss_out_file):

    print_steps = 100
    step = 0
    model.train()

    # seq: [seq length, embeddings]
    # labels: [label length, embeddings]
    with open(loss_out_file, 'a', encoding='utf-8') as ff:
        losses = []
        for seq, labels in train_set:
            step = step + 1

            # ignore the last word of the label scentence
            # because it is to be predicted
            label_input = labels[:-1].unsqueeze(0)

            # seq_input: [1, seq length, embeddings]
            seq_input = seq.unsqueeze(0)

            # y_pred: [1, seq length + 1, embeddings]
            y_pred = model(seq_input, label_input)

            # single_loss = loss_function(y_pred.squeeze(0), labels.to(self.device))
            single_loss = loss_function(y_pred.squeeze(0), labels.to(device))
            
            optimizer.zero_grad()
            single_loss.backward()
            optimizer.step()

            losses.append(single_loss.item())
            if print_steps != 0 and step%print_steps==1:
                print(f'[step: {step} - {time.asctime(time.localtime(time.time()))}] - loss:{single_loss.item():10.8f}')
                ff.write(f'{time.asctime(time.localtime(time.time()))},{str(batch_idx)},{str(step)},{np.mean(losses)}\n')
                losses.clear()

def do_batched_train(device, model:Seq2Seq, train_set, optimizer, loss_function, batches, print_steps, batch_idx, loss_out_file):
    losses = []
    step = 0

    model.train()

    x_input = torch.zeros(0, max_x_len, model.embeddings)
    y_input = torch.zeros(0, max_y_len - 1, model.embeddings)
    y_gold  = torch.zeros(0, max_y_len, model.embeddings)
    # seq: [seq length, embeddings]
    # labels: [label length, embeddings]
    
    with open(loss_out_file, 'a', encoding='utf-8') as ff:
        for seq, labels in train_set:

            if seq.shape[0] > max_x_len or labels.shape[0] > max_y_len:
                print('Scentence is too long. Ignored.' + str(seq.shape[0]) + " : " + str(labels.shape[0]))
                continue

            step = step + 1

            label_input = torch.zeros(1, max_y_len - 1, model.embeddings)
            gold_input = torch.zeros(1, max_y_len, model.embeddings)
            # ignore the last word of the label scentence
            # because it is to be predicted
            label_input[0,:labels.shape[0] - 1] = labels[:-1]
            gold_input[0, :labels.shape[0]] = labels
            y_input = torch.cat((y_input, label_input), dim=0)
            y_gold = torch.cat((y_gold, gold_input), dim=0)

            # seq_input: [1, seq length, embeddings]
            seq_input = torch.zeros(1, max_x_len, model.embeddings)
            seq_input[0, :seq.shape[0]] = seq
            x_input = torch.cat((x_input, seq_input), dim=0)

            if x_input.shape[0] == batches:
                loss = _do_batch_train(x_input.to(device), y_input.to(device), y_gold.to(device), model, optimizer, loss_function)
                
                losses.append(loss)
                if len(losses) >= print_steps:
                    ff.write(f'{time.asctime(time.localtime(time.time()))},{str(batch_idx)},{str(step)},{np.mean(losses)}\n')
                    print(f'[{time.asctime(time.localtime(time.time()))}] - {str(len(losses))}*{str(batches)}-avg:{np.mean(losses)}')
                    losses.clear()

                x_input = torch.zeros(0, max_x_len, model.embeddings)
                y_input = torch.zeros(0, max_y_len - 1, model.embeddings)
                y_gold  = torch.zeros(0, max_y_len, model.embeddings)

    if x_input.shape[0] > 0:
        _do_batch_train(x_input.to(device), y_input.to(device), y_gold.to(device), model, optimizer, loss_function)


def _do_batch_train(x_input, y_input, y_gold, model, optimizer, loss_function):
    # y_pred: [1, seq length + 1, embeddings]
    y_pred = model(x_input, y_input)

    loss = get_loss(y_pred, y_gold, model, optimizer, loss_function)
    return loss


def get_loss(predicted, gold, model, optimizer, loss_function):
    single_loss = loss_function(predicted, gold)
    optimizer.zero_grad()
    single_loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.2, norm_type=2)
    optimizer.step()
    loss = single_loss.item()
    return loss


def train(device, model, embedding_loader, corpus_fname, batch_size:int, batches: int):
    eos_en = tokenizer.EOS_EN
    eos_zh = tokenizer.EOS_ZH

    reader = corpus_reader.TmxHandler()
    loss = torch.nn.MSELoss()
    # loss = torch.nn.L1Loss()

    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = torch.optim.Adam(model.parameters())
    
    generator = reader.parse(corpus_fname)
    for _b in range(batches):
        batch = []
        try:
            for _ in range(batch_size):
                try:
                    corpus = next(generator)
                    if 'en' in corpus and 'zh' in corpus:
                        # en = embedding_loader.get_scentence_embeddings(corpus['en'] + eos_en, 'en').to(device)
                        # zh = embedding_loader.get_scentence_embeddings(corpus['zh'] + eos_zh, 'zh').to(device)
                        en = embedding_loader.get_scentence_embeddings(corpus['en'], 'en').to(device)
                        zh = embedding_loader.get_scentence_embeddings(corpus['zh'], 'zh').to(device)
                        batch.append((en,zh))
                except (StopIteration):
                    break
        finally:
            print("batch: " + str(_b))
            do_batched_train(device, model, batch, optimizer, loss, 2, 50, _b, "../loss.log")
            # do_train(device, model, batch, optimizer, loss, _b, "../loss.log")
            model_out_name = "./models/seq2seq_" + str(time.time())
            print(f'[{time.asctime(time.localtime(time.time()))}] - saving model:{model_out_name}')
            torch.save(model, model_out_name)
        
if __name__=="__main__":
    device = torch.device('cuda')
    embeddings = 300
    hiddens = 1200
    n_layers = 4

    print("loading embedding")
    embedding_loader = word2vec.WordEmbeddingLoader("../embeddings/sgns.merge/sgns.merge.word")
    # embedding_loader = word2vec.WordEmbeddingLoader("../embeddings/parellel_01.v2c")
    print("load embedding finished")

    # model_fname = "./models/_seq2seq_1698000846.3281412"
    model_fname = None
    model = None
    if not model_fname is None:
        print('loading model from ' + model_fname)
        model = torch.load(model_fname, map_location=device)
        print('model loaded')
    else:
        model = Seq2Seq(device, embeddings, hiddens, n_layers, 0.5, 0.5).to(device)
    
    train(device, model, embedding_loader, "../corpus/train", 5000, 4000)