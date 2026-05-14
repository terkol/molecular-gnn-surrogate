import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from model import Network
from data_processing import create_graph_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(create_graph_dataset('zinc250k_selfies.csv'), shuffle=True)

model = Network(num_features=6).to(device)

optimizer = torch.optim.Adam(model.parameters())

model.train()
for epoch in range(200):
    total_loss = 0

    for batch in loader:
        optimizer.zero_grad()
        batch = batch.to(device)
        
        prediction = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(prediction, batch.y)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
    # print loss every 10 epochs
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')