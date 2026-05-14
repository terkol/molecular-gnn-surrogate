import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_add_pool

class Network(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        # Using three convolutional layers since apperently that's best
        self.conv1 = GCNConv(num_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
    
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 1)

    def forward(self, x, edge_index, batch):
        # Using relu activation 
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        x = global_add_pool(x, batch)
        
        x = F.relu(self.linear1(x))
        return self.linear2(x)