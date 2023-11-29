import torch.nn as nn
import torch

EPSILON = -1e10
        
def create_mask(lengths):
    # mask = torch.arange(length.max())[None, :] < lengths[:, None]
    mask = torch.arange(lengths.max())[None, :, None].to(lengths.device) < lengths[:,None, None]
    return mask

class Attention(nn.Module):
    def __init__(self, device, encoded_hiddens, decoded_hiddens):
        super().__init__()
        self.device = device
        self.encoded_size = encoded_hiddens
        self.decoded_size = decoded_hiddens
        
        self.enc_linear = nn.Linear(encoded_hiddens, decoded_hiddens, bias=False)
        self.dec_linear = nn.Linear(decoded_hiddens, decoded_hiddens, bias=False)
        
        self.softmax = nn.Softmax(-1)
        
        nn.init.xavier_normal_(self.enc_linear.weight.data)
        nn.init.xavier_normal_(self.dec_linear.weight.data)

    # h:     [batches, max_h_seq_length, encoder_hiddens]
    # h_len: [batches]
    # y:     [batches, max_y_seq_length, decoder_hiddens]
    # y_len: [batches]
    def forward(self, h, h_len, y, y_len):
        x_mask = create_mask(h_len).to(h.device)
        y_mask = create_mask(y_len).to(y.device)
        mask = torch.bmm(y_mask.float(), x_mask.float().transpose(1,2))
        enc = self.enc_linear(h)
        dec = self.dec_linear(y)
        attn = torch.bmm(dec,enc.transpose(1,2))
        attn = attn * mask
        attn = self.softmax(attn + EPSILON)
        return attn
        
if __name__ == "__main__":
    dev = torch.device("cpu")
    encoded_hiddens = 4
    decoded_hiddens = 3
    
    attn = Attention(dev, encoded_hiddens, decoded_hiddens)
    
    h = torch.Tensor([[[1,1,1,1],[1,1,1,1],[1,1,1,1]],[[2,2,2,2],[2,2,2,2],[2,2,2,2]]])
    h_len = torch.Tensor([3,2])
    y = torch.Tensor([[[1,1,1],[1,1,1],[1,1,1],[1,1,1]],[[2,2,2],[2,2,2],[2,2,2],[2,2,2]]])
    y_len = torch.Tensor([4,3])
    
    at = attn(h, h_len, y, y_len)
    print(at)