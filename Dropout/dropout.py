import numpy as np
import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self,p = 0.5):
        super().__init__()
        self.p = p

    def forward(self,x):
        if not self.training or self.p == 0:
            return x

        mask = (torch.rand(x.shape) > self.p).float()

        return (x * mask) / (1 - self.p)


drop = Dropout(0.5)
x = torch.tensor([10., 20., 30., 40.])

drop.train()
print("Training:", drop(x))

drop.eval()
print("Testing:", drop(x))