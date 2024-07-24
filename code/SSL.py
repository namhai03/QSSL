import torch
import tqdm
import torch.nn as nn
from einops import repeat

class CreateMemoryBank():
    def __init__(self, model, trainqueue, batch_size, device):
        self.model = model
        self.trainqueue = trainqueue
        self.batch_size = batch_size
        self.device = device
    def initialize(self):
        self.model.eval()
        print('CREATE MemoryBank: {}'.format(len(self.trainqueue)))
        memorybank = []
        for i , (data, _) in tqdm.tqdm(enumerate(self.trainqueue)):
            x = self.model(data.to(self.device))
            memorybank.append(x)
        return memorybank

def KLLoss(x, y, pos, neg, gamma):
    neg_loss = 0.
    criterion = nn.KLDivLoss(reduction='sum', log_target = True)
    pos_loss = criterion(pos,x) + criterion(pos,y)
    pos_loss = pos_loss/x.size(0)
    for i in range(x.size(0)):
        x = repeat(x[i].unsqueeze(0), '1 d -> h d', h = neg.size(0))
        y = repeat(y[i].unsqueeze(0), '1 d -> h d', h = neg.size(0))
        neg_loss = neg_loss + (criterion(neg,x) + criterion(neg, y))/neg.size(0)
    neg_loss = neg_loss/x.size(0)
    return gamma*pos_loss + (1-gamma)*neg_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
