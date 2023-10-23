import torch
import encoder as enc
import decoder as dec
import torch.nn as nn
import torch.nn.functional as F
import time

class Seq2Seq(nn.Module):
    def __init__(self, device, embeddings, hiddens, n_layers):
        super().__init__()
        self.device = device
        self.encoder = enc.Encoder(device, embeddings, hiddens, n_layers).to(device)
        self.decoder= dec.Decoder(device, embeddings, hiddens, n_layers).to(device)
        self.embeddings = self.encoder.embedding_size
        assert self.encoder.n_layers == self.decoder.n_layers, "Number of layers of encoder and decoder must be equal!"
        assert self.decoder.hidden_layer_size==self.decoder.hidden_layer_size, "Hidden layer size of encoder and decoder must be equal!"

    # x = [x length, embeddings]
    # y = [batches, y length, embeddings]
    def forward(self, x, y):

        # hidden, cell = [n layers, batch size, embeddings]
        x = x.unsqueeze(0).to(self.device)
        hidden, cell = self.encoder(x)

        batches = y.shape[0]
        # index = y.shape[1]

        # now hidden and cell is [batch_size, x length, embeddings]
        hidden = torch.tile(hidden, dims=(1, batches, 1)).to(self.device)
        cell = torch.tile(cell, dims=(1, batches, 1)).to(self.device)
        decoder_input = y.to(self.device)

        # predicted = [batches, y length, embeddings]
        predicted, hidden, cell = self.decoder(decoder_input, hidden, cell)

        return predicted


    def do_train(self, train_inout_seq, optimizer, loss_function, print_steps=100, epoches=1):
        now = time.time()

        for epoch in range(epoches):
            self.train()
            # seq = [seq length, embeddings]
            # labels = [label length, embeddings]
            step = 0
            for seq, labels in train_inout_seq:
                step = step + 1

                seq = seq.to(self.device)

                # batches = labels.shape[0] + 1
                # for i in range(batches):
                #     label_input=torch.zeros(i + 1, self.embeddings)
                #     for j in range(1, i + 1):
                #         label_input[j] = labels[j-1]

                #     label_input = label_input.unsqueeze(0).to(self.device)
                #     y_pred = model(seq, label_input)

                #     label_teacher=torch.zeros(i + 1, self.embeddings)
                #     for j in range(1, i + 1):
                #         label_teacher[j-1] = labels[j-1]
                label_input = torch.zeros(labels.shape[0], self.embeddings)
                label_input[1:] = labels[:-1]
                label_input = label_input.unsqueeze(0).to(self.device)
                y_pred = self(seq, label_input)

                single_loss = loss_function(y_pred.squeeze(0), labels.to(self.device))
                
                optimizer.zero_grad()
                single_loss.backward()
                optimizer.step()

                if print_steps != 0 and step%print_steps==1:
                    print(f'epoch: {step:3} loss: {single_loss.item():10.8f}')

        # print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')
        torch.save(self, "./models/seq2seq_" + str(now))
    
