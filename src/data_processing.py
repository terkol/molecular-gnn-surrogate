from pathlib import Path
import pandas as pd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from rdkit.Chem import Descriptors

def get_atom_features(atom):
    """
    Get mass, connections (other than hydrogen), formal charge, aromaticity, number of hydrogens and hybridization state
    """
    features = [atom.GetMass(), atom.GetDegree(), atom.GetFormalCharge(), int(atom.GetIsAromatic()), atom.GetTotalNumHs(), atom.GetHybridization()]
    return torch.tensor(features, dtype=torch.float)

def create_graph_dataset(file_name, sample_size=None):
    """
    Parses SMILES strings into PyTorch Geometric Data objects
    """ 
    path = Path(__file__).parent
    df = pd.read_csv(path / 'data' / file_name)

    smiles_list = df.smiles.head(10).to_list()
    data_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: 
            continue
        x = torch.stack([get_atom_features(atom) for atom in mol.GetAtoms()])
        edges = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edges.extend([[i,j],[j,i]])

        edge_index = torch.tensor(edges, dtype=torch.long).t()
        y = torch.tensor([[Descriptors.MolLogP(mol)]], dtype=torch.float)
        data_list.append(Data(x=x, edge_index=edge_index, y=y))

    return data_list