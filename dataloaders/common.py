import os
import warnings
import random
import pandas as pd
import torch
import numpy as np
import pymatgen
from tqdm import tqdm

from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from models.cgcnn_atom_features import atom_features

def generate_site_species_vector(structure: pymatgen.core.structure.Structure, ATOM_NUM_UPPER):

    if hasattr(structure, 'species'):
        atom_pos = torch.tensor(structure.cart_coords, dtype=torch.float)
        atom_num = torch.tensor(structure.atomic_numbers, dtype=torch.long).unsqueeze_(-1)
        x_species_vector = torch.eye(ATOM_NUM_UPPER)[atom_num - 1].squeeze()

    else:
        x_species_vector = []
        for site in structure.species_and_occu:
            site_species_and_occupancy = []
            # For each element at the site, get one-hot encoding and multiply the site occupancy to calculate the element occupancy vector.
            for elem in site.elements:
                if type(elem) == pymatgen.core.Element:
                    occupancy = site.element_composition[elem]
                elif type(elem) == pymatgen.core.periodic_table.Specie or type(elem) == pymatgen.core.periodic_table.Species:
                    occupancy = site.element_composition[elem.element]
                elif type(elem) == pymatgen.core.composition.Composition:
                    occupancy = site.element_composition[elem.element]
                    # print(elem, occupancy)
                elif type(elem) == pymatgen.core.periodic_table.DummySpecie or type(elem) == pymatgen.core.periodic_table.DummySpecies:
                    raise ValueError(f'Unsupported specie: {site}! Skipped')
                else:
                    print(site, type(site))
                    raise AttributeError
                atom_num = torch.tensor(elem.Z, dtype=torch.long)
                elem_onehot = torch.eye(ATOM_NUM_UPPER)[atom_num - 1]
                site_species_and_occupancy.append(elem_onehot*occupancy)
            # Sum of one-hot vector for each element at the site and convert to site occupancy
            site_species_and_occupancy_sum = torch.stack(site_species_and_occupancy).sum(0)
            x_species_vector.append(site_species_and_occupancy_sum)
        x_species_vector = torch.stack(x_species_vector, 0)
        
    return x_species_vector

def exclude_one_atom_crystal(data):
    # Set the default n > 1. This is to ensure that
    # when data has neither pos nor x (eg, xrd data)
    # the code returns True (ie, not exclude).
    n = 2
    if hasattr(data, 'pos') and data.pos is not None:
        n = data.pos.shape[0]
    elif hasattr(data, 'x') and data.x is not None:
        n = data.x.shape[0]

    if n > 1:
        return True

    return False

def apply_pre_filters(data, conditions):
    """
    Applies all conditions to the `data` and performs an AND operation.

    :param data: The PyG data object to be examined.
    :param conditions: A list of functions to apply.
    :return: True if all conditions are satisfied, False otherwise.
    """
    for condition in conditions:
        if not condition(data):
            return False
    return True

def exclude_unk_titles(data, tokenizer):
    # memo: tokenizer.unk_token_id for 'allenai/scibert_scivocab_uncased' is 101
    if tokenizer.unk_token_id in data.tokenized_title["input_ids"]:
        return False
    else:
        return True

def read_structure_from_cif(cifpath):  
    try:  
        return Structure.from_file(cifpath, primitive=False)  
    except (ValueError, AssertionError) as e:  
        print(e)  
        print(f'file: {cifpath} has been skipped')  
        return None  
  
def generate_full_path(filename: str, base_path: str = "/cod") -> str:
    """Generate full path for a CIF file based on COD structure.
    
    Args:
        filename: COD ID (must be at least 6 characters)
        base_path: Base directory path for COD
        
    Returns:
        str: Full path to the CIF file
        
    Examples:
        >>> generate_full_path("1000001", "/cod")
        '/cod/1/00/00/1000001.cif'
        >>> generate_full_path("2345678", "/data")
        '/data/2/34/56/2345678.cif'
        >>> generate_full_path("1234567890", "/cod")
        '/cod/1/23/45/1234567890.cif'
        >>> try:
        ...     generate_full_path("12345", "/cod")
        ... except ValueError as e:
        ...     print("Error caught")
        Error caught
    """
    if len(filename) < 6:
        raise ValueError("ファイル名は6文字以上である必要があります")

    first_part = filename[0]
    second_part = filename[1:3]
    third_part = filename[3:5]

    return f"{base_path}/{first_part}/{second_part}/{third_part}/{filename}.cif"


def get_material_properties(cifpath, structure):  
    if structure is None:  
        return {"final_structure": None, "file_id": cifpath.stem, "formula": None}  
    return {"final_structure": structure, "file_id": cifpath.stem, "formula": structure.formula}  
  
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)