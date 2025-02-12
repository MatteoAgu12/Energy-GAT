from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.analysis.bond_valence import BVAnalyzer
from torch_geometric.data import Data
from typing import List, Dict
import numpy as np
import torch
import pickle

periodic_table = {
    "H": 1, "He": 2,
    "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26,
    "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31, "Ge": 32, "As": 33, "Se": 34,
    "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58,
    "Pr": 59, "Nd": 60, "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73, "W": 74,
    "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82,
    "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98,
    "Es": 99, "Fm": 100, "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105,
    "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112,
    "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

def create_check_file(material: Dict, 
                      GW_correction: float, DFT_gap: float,
                      path: str = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\gap_check.txt') -> None:
    material_name = ''
    
    for key, value in material['atoms'].items():
        degen = len(value)
        if degen > 1:
            material_name = material_name + str(degen) + str(key)
        elif degen == 1:
            material_name = material_name + str(key)
        
    try:
        with open(path, 'a') as file:
            file.write(f"{material_name} {GW_correction + DFT_gap} {DFT_gap} {GW_correction}\n")
                
    except Exception as e:
        print(f"Error in updating the file: {e}")
    
    return

def create_graph_data_interpol(material: Dict, update_check: bool=False) -> Data:
    # Node features: concatenate atomic number and position
    atoms = []
    positions = []
    elements = material['atoms'].keys()
    sites = material['atoms'].values()
    
    # Create the array of atomic numbers
    for element in elements:
        degen = len(material['atoms'][element])
        for i in range(degen):
            atoms.append(periodic_table[element])
            
    # Create the matrix of positions
    for atom in sites:
        for site in atom:
            positions.append(site)
            
    if len(atoms) is not len(positions):
        raise ValueError("Atoms and positions don't have the same lenght.")
                
    sorted_indices = sorted(range(len(atoms)), key=lambda i: atoms[i])
    atoms = [atoms[i] for i in sorted_indices]
    positions = [positions[i] for i in sorted_indices]
    
    node_features = torch.cat([
        torch.tensor(atoms, dtype=torch.float32).view(-1, 1),
        torch.tensor(positions)
    ], dim=1)

    # Edges: fully connected graph (you can improve this by using bonding information if available)
    num_nodes = len(atoms)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it bidirectional

    # Global features (used as part of node features or separately)
    gap_DFT = material['gap_DFT']
    crystal_params = [ material['A'][0], material['A'][1], material['A'][2], 
                       material['B'][0], material['B'][1], material['B'][2], 
                       material['C'][0], material['B'][1], material['C'][2],
                       material['alpha'], material['beta'], material['gamma'], 
                       material['volume'] ]
    
    # Combine valence and conduction band energies into a single tensor
    valence_band_energies = torch.tensor(material['DFT_valence_band'])
    conduction_band_energies = torch.tensor(material['DFT_conduction_band'])
    band_energies = torch.cat([valence_band_energies, conduction_band_energies], dim=0)  # Shape: (2 * num_bands)
    
    # Target
    correction = material['GW_correction']

    # Create the graph data object
    data = Data(
        x=node_features, 
        edge_index=edge_index, 
        y=torch.tensor([correction], dtype=torch.float32),
        band_energies=band_energies.float(),
        crystal_params=torch.tensor(crystal_params, dtype=torch.float)
    )
    
    # Upload the check file
    if update_check:
        create_check_file(material, correction, gap_DFT)
        
    return data

def create_graph_data_K(material: Dict, update_check: bool = False) -> Data:
    # Node features: concatenate atomic number and position
    atoms = []
    positions = []
    elements = material['atoms'].keys()
    sites = material['atoms'].values()
    
    # Create the array of atomic numbers
    for element in elements:
        degen = len(material['atoms'][element])
        for i in range(degen):
            atoms.append(periodic_table[element])
            
    # Create the matrix of positions
    for atom in sites:
        for site in atom:
            positions.append(site)
            
    if len(atoms) is not len(positions):
        raise ValueError("Atoms and positions don't have the same lenght.")
                
    sorted_indices = sorted(range(len(atoms)), key=lambda i: atoms[i])
    atoms = [atoms[i] for i in sorted_indices]
    positions = [positions[i] for i in sorted_indices]
    
    node_features = torch.cat([
        torch.tensor(atoms, dtype=torch.float32).view(-1, 1),
        torch.tensor(positions)
    ], dim=1)

    # Edges: fully connected graph (you can improve this by using bonding information if available)
    num_nodes = len(atoms)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # Make it bidirectional

    # Global features (used as part of node features or separately)
    gap_DFT = material['gap_DFT']
    crystal_params = [ material['A'][0], material['A'][1], material['A'][2], 
                       material['B'][0], material['B'][1], material['B'][2], 
                       material['C'][0], material['B'][1], material['C'][2],
                       material['alpha'], material['beta'], material['gamma'], 
                       material['volume'] ]
    
    # Combine valence and conduction band energies into a single tensor
    valence_band_energies = torch.tensor(material['DFT_valence_band'])
    conduction_band_energies = torch.tensor(material['DFT_conduction_band'])
    band_energies = torch.cat([valence_band_energies, conduction_band_energies], dim=0)  # Shape: (2 * num_bands)
    
    # k-points
    k_points = torch.tensor(material['k_points'], dtype=torch.float)
    
    # Target
    correction = material['GW_correction']

    # Create the graph data object
    data = Data(
        x=node_features, 
        edge_index=edge_index, 
        y=torch.tensor([correction], dtype=torch.float32),
        band_energies=band_energies.float(),
        crystal_params=torch.tensor(crystal_params, dtype=torch.float),
        k_points = k_points
    )
    
    # Upload the check file
    if update_check:
        create_check_file(material, correction, gap_DFT)
        
    return data

if __name__ == '__main__':
    path_interpol = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_interpol.pkl'
    with open(path_interpol, "rb") as file:
        DB_interpol = pickle.load(file)
        
    path_K = r'C:\Users\aguia\OneDrive\Desktop\Physics_I_anno\Computational_Material_Physics\Project\Final\TrainSet\DB_K.pkl'
    with open(path_K, "rb") as file:
        DB_K = pickle.load(file)
       
    graphs_interpol = []
    graphs_K = []
    for mat in DB_interpol:
        graph = create_graph_data_interpol(mat)
        graphs_interpol.append(graph)
    for mat in DB_K:
        graph = create_graph_data_interpol(mat)
        graphs_K.append(graph)