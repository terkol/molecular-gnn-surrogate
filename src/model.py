import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool
    
class Network(nn.Module):
    def __init__(self, f):
        super().__init__()
        # Using three convolutional layers since apperently that's best
        self.c1 = GCNConv(f, 64)
        self.c2 = GCNConv(64, 64)
        self.c3 = GCNConv(64, 64)
        
        self.l1 = nn.Linear(64, 32)
        self.l2 = nn.Linear(32, 1)

    def forward(self, x, e, b):
        # Using relu activation 
        x = F.relu(self.c1(x, e))
        x = F.relu(self.c2(x, e))
        x = self.c3(x, e)

        x = global_add_pool(x, b)
        
        return self.l2(F.relu(self.l1(x)))