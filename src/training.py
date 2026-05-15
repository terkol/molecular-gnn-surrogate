import torch
import torch.nn.functional as F
from torch_geometric.loader import *
from model import *
from data_processing import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(get_data('zinc250k_selfies.csv'))

model = Network(6).to(device)
optimizer = torch.optim.Adam(model.parameters())

model.train()
for epoch in range(200):
    total_loss = 0
    for mol in loader:
        optimizer.zero_grad()
        mol = mol.to(device)
        
        preds = model(mol.x, mol.edge_index, mol.batch)
        loss = F.mse_loss(preds, mol.y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    if epoch % 10 == 0: # print loss every 10 epochs
        print(f"Epoch: {epoch} | Loss: {total_loss / len(loader)}") 