from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_features(a):
    return torch.tensor([a.GetMass(), a.GetDegree(), a.GetFormalCharge(), int(a.GetIsAromatic()), a.GetTotalNumHs(), a.GetHybridization()], dtype=torch.float)

def get_data(f):
    """
    Currently only takes the first ten smiles strings forward, just 
    to validate the pipeline.

    Input:
    ------
        File path to a csv file with a column called 
        'smiles' that contains the smiles strings.

    Output:
    ------
        List of PyG Data objects, one for each smiles string. 
    """
    df = pd.read_csv(Path(__file__).parent.parent / "data" / f)
    
    # just do 10 for now
    smiles_strings = df.smiles.head(10).to_list()
    data = []
    
    for smiles in smiles_strings:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: continue
        
        x = torch.stack([get_features(a) for a in mol.GetAtoms()])
        bonds = []
        
        for b in mol.GetBonds():
            i = b.GetBeginAtomIdx()
            j = b.GetEndAtomIdx()
            bonds.extend([[i,j],[j,i]]) # Since PyG graphs are undirected

        ei = torch.tensor(bonds, dtype=torch.long).t()
        y = torch.tensor([[Descriptors.MolLogP(mol)]], dtype=torch.float)
        
        data.append(Data(x=x, edge_index=ei, y=y))

    return data