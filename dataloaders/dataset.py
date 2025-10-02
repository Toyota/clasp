#!/usr/bin/env python
# coding: utf-8
from tqdm import tqdm
from pathlib import Path
from typing import Callable
from multiprocessing import Pool
from functools import partial
import multiprocessing
import os.path as osp
import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from dataloaders.common import exclude_one_atom_crystal, exclude_unk_titles, apply_pre_filters, read_structure_from_cif, generate_full_path, get_material_properties
from dataloaders.dataset_cgcnn import ExportCrystalGraph, make_data


class ClaspDataset(InMemoryDataset):    
    def __init__(self, input_dataframe: pd.DataFrame, 
                 tokenizer: Callable[[str], list], 
                 max_token_length=128,
                 root="data/", atom_feat_mode='original', max_num_nbr=12,
                 radius=8, dmin=0, step=0.2):    
        self.ATOM_NUM_UPPER = 98    
        self.input_dataframe = input_dataframe    
        self.tokenizer = tokenizer  # Tokenizer function passed as an argument
        self.max_token_length = max_token_length
        self.use_primitive = False    
        self.atom_fea_original = atom_feat_mode == 'original'    
        self.cg_exporter = ExportCrystalGraph(atom_feat_mode, max_num_nbr, radius, dmin, step)    
        self.process_chunk_size = 5000
        exclude_unk_titles_partial = partial(exclude_unk_titles, tokenizer=tokenizer)
        apply_pre_filters_partial = partial(apply_pre_filters, conditions=[exclude_one_atom_crystal, exclude_unk_titles_partial])
        super(ClaspDataset, self).__init__(root, pre_filter=apply_pre_filters_partial)   
        self.load(self.processed_paths[0]) 
        # self.data, self.slices = torch.load(self.processed_paths[0])   
  
    @property  
    def raw_file_names(self):  
        return "raw"  
  
    def download(self):  
        pass  

    def convert_material_to_PyGgraph(self, material):    
        try:    
            assert material['file_id'] is not None    
            assert material['formula'] is not None    
            assert material['final_structure'].num_sites <=500, "structure has over 500 sites! skipped"
        
            data = make_data(material, self.cg_exporter, self.use_primitive)
            if data is None:    
                return None    
        
            data.material_id = material['file_id']    
            data.pretty_formula = material['formula']    
            return data    
        except (AssertionError, AttributeError, 
                IndexError, ValueError, TypeError) as e:    
            print(e)    
            # print(f"material id: {material['file_id']}")    
            return None   
    
    def save_chunk(self, data_chunk, filename):
        torch.save(data_chunk, filename)

    def load_and_combine_chunks(self, chunk_filenames):
        combined_data = []
        for filename in chunk_filenames:
            data_chunk = torch.load(filename)
            combined_data += data_chunk
        return combined_data
    
    def process_individual(self, arg):
        cif_file, title = arg
        structure = read_structure_from_cif(cif_file)    
        material = get_material_properties(cif_file, structure)
        material["title"] = title  # Original title: str

        material["tokenized_title"] = self.tokenizer(title, 
                                                        return_tensors="pt", 
                                                        max_length=self.max_token_length, 
                                                        padding="max_length",
                                                        truncation=True)
        
        data = self.convert_material_to_PyGgraph(material)
        
        if self.pre_transform is not None:    
            data = self.pre_transform(data)    

        return data
  

    def process(self):    
        crystals = self.input_dataframe["cif_path"].apply(Path).tolist()
        titles = self.input_dataframe["title"]
        print('loaded data: ', self.raw_paths[0])    

        args = list(zip(crystals, titles))

        chunk_filenames = []
        with Pool(int(multiprocessing.cpu_count()/2)) as pool:
            for i in tqdm(range(0, len(args), self.process_chunk_size)):
                chunk = args[i:i+self.process_chunk_size]

                # Process the current chunk
                results = pool.imap_unordered(self.process_individual, chunk, 100)
                # Explicitly filter out None values if they exist.
                data_chunk = [data for data in results if data is not None]

                if self.pre_filter is not None:    
                    data_chunk = [data for data in data_chunk if self.pre_filter(data)]

                # Save the processed chunk to a temporary file
                chunk_filename = f'{self.processed_dir}/temp_chunk_{i//self.process_chunk_size}.pt'
                self.save_chunk(data_chunk, chunk_filename)
                chunk_filenames.append(chunk_filename)

        # After all chunks have been processed, load them back and combine
        data_list = self.load_and_combine_chunks(chunk_filenames)

        # Clean up temporary chunk files
        for filename in chunk_filenames:
            Path(filename).unlink()

        self.save(data_list, self.processed_paths[0])
        # data, slices = self.collate(data_list)    
        # torch.save((data, slices), self.processed_paths[0])


    @property
    def processed_file_names(self):
        suf = "" if self.atom_fea_original else "_pn"
        if self.use_primitive:
            return f'processed_data_cgcnn{suf}.pt'
        else:
            return f'processed_data_convcell_cgcnn{suf}.pt'