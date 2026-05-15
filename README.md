# Molecular Graph Neural Network Surrogate

This repository implements a Physics-Informed Graph Convolutional Network (GCN) designed to act as a high-speed surrogate model for predicting molecular properties (specifically LogP) directly from text-based SMILES strings. 

The pipeline bridges raw cheminformatics data processing with deep learning, bypassing computationally expensive physics-based simulations by utilizing a Message Passing Neural Network (MPNN).

## Architecture

The codebase is decoupled into distinct data processing and execution logic, following industrial MLOps standards.

### Data Ingestion & Transformation (`src/data_processing.py`)

Parses SMILES strings into spatial graphs with RDkit. Extracts fundamental atomic features (mass, degree, formal charge, aromaticity, hybridization) into a continuous feature tensor and outputs everything into PyTorch Geometric Data objects.


### Network Topology (`src/model.py`)

Utilizes a 3-Layer Graph Convolutional Network (GCNConv) and ReLU activation. Global add pooling integrates localized atomic states into a unified molecular representation before final linear regression.

### Execution Engine (`src/training.py`)

Runs the main training process from start to finish. The script trains the model for 200 epochs, using the Adam optimizer and Mean Squared Error (MSE) loss to improve prediction accuracy. To track progress, it prints the average training loss every 10 epochs so you can verify the model is learning correctly.

## Repository Structure

    molecular-gnn-surrogate/
    ├── data/
    │   └── zinc250k_selfies.csv      # Smiles strings, selfies etc.
    ├── src/
    │   ├── data_processing.py        # Graph topology and feature extraction
    │   ├── model.py                  # PyTorch Geometric neural architecture
    │   └── training.py               # Execution and validation loop
    ├── requirements.txt
    └── README.md                     

## Execution Instructions

 Create the environment:

`conda create -n gnn-surrogate python=3.10`

Activate the environment:

`conda activate gnn-surrogate`

Install the required dependencies

`pip install -r requirements.txt`

Execute `training.py`:

`python src/training.py`