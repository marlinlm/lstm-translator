import torch
import torch.nn as nn

class Creterion(nn.Module):
    
    def __init__(self, dev):
        super().__init__()
        self.device = dev
    
    def forward(self, predicted, target, target_len, batches):
        # pred_flat = predicted.view(-1, predicted.shape[2])
        # target_flat = target.view(-1, 1).long()
        
        mask = (torch.arange(target_len.max().item(), device=self.device)[None, :] < target_len[:, None]).float()
        loss = predicted.gather(2, target.long().view(target.shape[0], -1, 1))
        loss = torch.log(loss.view(-1, 1)) * mask.view(-1, 1)
        # loss = loss.view(-1, 1) * mask.view(-1, 1)
        loss = - torch.sum(loss) * batches / torch.sum(mask)
        return loss
        
if __name__=="__main__":
    dev = torch.device("cpu")
    loss = Creterion(dev)
    p = torch.Tensor([[[0.1, 0.2, 0.3, 0.4],[0.2,0.2,0.3,0.3],[0.2,0.2,0.3,0.3]],[[0.2,0.5,0.1,0.2],[0.3,0.2,0.4,0.1],[0.2,0.2,0.3,0.3]]]).to(dev)
    t = torch.Tensor([[2,1],[3,2]]).to(dev)
    length = torch.Tensor([2,1])
    print(loss(p,t,length))
    