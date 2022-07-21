import torch
import torch.nn as nn
import torch.nn.functional as F

class DFREncoder(nn.Module):
    def __init__(self, w_size, p_size):
        super(DFREncoder, self).__init__()
        self.w_size = w_size
        self.hidden_size = 1024
        self.p_size = p_size
        self.fc1 = nn.Linear(self.w_size, self.hidden_size, bias=False)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.fc3 = nn.Linear(self.hidden_size, self.p_size, bias=False)
    
    def forward(self, x):
        x = x.view(-1, 18*512)
        out = F.elu(self.fc1(x), alpha=0.1)
        out = F.elu(self.fc2(out), alpha=0.1)
        out = self.fc3(out)
        return out

if __name__ == '__main__':
    encoder = DFREncoder(18*512, 257)
    w = torch.rand(3, 18, 512)
    out = encoder(w)
    print(out.shape)
        




