import pandas as pd 
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

def get_dataset_seq_lengths(dataset, leq):
    length_dict = dict()
    for sample in dataset:
        seq_leng = len(sample[1]['protein']['sequence'])

        if seq_leng in length_dict.keys() and seq_leng <= leq:
            length_dict[seq_leng] += 1
        elif seq_leng <= leq:
            length_dict[seq_leng] = 1

    sorted_dict = dict(sorted(length_dict.items(), key=lambda item: item[0]))
    return sorted_dict

def get_subset_leq_len(dataset, leq):
    data_subset = []
    for sample in dataset:
        seq_leng = len(sample[1]['protein']['sequence'])
        if seq_leng <= leq:
            data_subset.append(sample)
    return data_subset

def get_max_seq_len(dataset):
    max_len = 0
    for sample in tqdm(dataset):
        seq_leng = len(sample[1]['protein']['sequence'])
        if seq_leng > max_len:
            max_len = seq_leng
    return max_len

def filter_by_max_length_and_pad(dataset, max_seq_len, return_proteins = False):
    proteins = []
    data_list = []

    for sample in dataset:
        data_list.append(sample[0][:,:3])

    xyz_mean = torch.mean(torch.concatenate(data_list, dim = 0), dim = 0)
    xyz_std = torch.std(torch.concatenate(data_list, dim = 0), dim = 0)



    if return_proteins:
        org_protein_data = []
        for sample in tqdm(dataset):
            seq_leng = len(sample[1]['protein']['sequence'])
            if seq_leng <= max_seq_len:
                padded_data = pad_cloud_data(sample[0], max_seq_len)
                padded_data[:,:3] = (padded_data[:,:3] - xyz_mean) / xyz_std
                proteins.append(padded_data)
                org_protein_data.append(sample)

        return proteins, org_protein_data
    
    else:
        for sample in tqdm(dataset):
            seq_leng = len(sample[1]['protein']['sequence'])
            if seq_leng <= max_seq_len:
                padded_data = pad_cloud_data(sample[0], max_seq_len)
                padded_data[:,:3] = (padded_data[:,:3] - xyz_mean) / xyz_std
                proteins.append(padded_data)
        return proteins

    
    
def pad_cloud_data(sample, max_seq_len):
    coords = sample[:,:3]
    coords = center_point_cloud(coords)

    labels = sample[:,3]
    coords = torch.nn.functional.pad(coords[:max_seq_len], (0,0,0,max(0, max_seq_len - coords.shape[0])))
    labels = torch.nn.functional.pad(labels[:max_seq_len], (0, max(0, max_seq_len - labels.shape[0])), value=20).unsqueeze(1)
    padded = torch.concatenate([coords, labels], dim = 1)
    return padded

def center_point_cloud(pc):
    center = torch.mean(pc, dim = 0)
    centered_pc = pc - center
    return centered_pc

def scale_cloud_data(pc):
    scaling_const = torch.sqrt(torch.max(torch.sum(pc**2), dim = -1).values)
    pc = pc / scaling_const
    return pc

def one_hot_encode_seq(seq, max_seq_len, transformer_input = False, convert_to_tensor=True):

    amino_encoding_dict = {'A': 0,'R': 1,'N': 2,'D': 3,'C': 4,'E': 5,
                            'Q': 6,'G': 7,'H': 8,'I': 9,'L': 10,'K': 11,
                            'M': 12,'F': 13,'P': 14,'S': 15,'T': 16,
                            'W': 17,'Y': 18,'V': 19, 'X':20}
    
    one_hot_idx_list = []
    one_hot_encoded_list = []
    seq = seq+(max_seq_len-len(seq))*'X'


    for char in seq:
        one_hot_idx_list.append([amino_encoding_dict[char]])
        one_hot = np.zeros(21)
        one_hot[amino_encoding_dict[char]] = 1

        one_hot_encoded_list.append(one_hot)

    encoded_idx_array = np.array(one_hot_idx_list)
    one_hot_encoded_array = np.array(one_hot_encoded_list)

    if transformer_input:
        return torch.tensor(encoded_idx_array, dtype=torch.int32).squeeze(), torch.tensor(one_hot_encoded_array, dtype=torch.float32)
    else:
        return torch.tensor(one_hot_encoded_array, dtype=torch.float32)
    

def make_one_hot_data_list(dataset, max_seq_len, transformer_input, return_proteins = True):
    one_hot_dataset = []
    proteins = []

    for sample in tqdm(dataset):

        if len(sample[1]['protein']['sequence']) > max_seq_len:
            continue

        else:

            if return_proteins:
                proteins.append(sample)

            if transformer_input:
                encoded_idx_tensor, one_hot_encoded_tensor = one_hot_encode_seq(sample[1]['protein']['sequence'], max_seq_len, transformer_input)
                one_hot_dataset.append((encoded_idx_tensor, one_hot_encoded_tensor))

            else:
                one_hot_seq = one_hot_encode_seq(sample[1]['protein']['sequence'], max_seq_len, transformer_input)
                one_hot_dataset.append(one_hot_seq)

    if return_proteins:

        return one_hot_dataset, proteins
    else:
        return one_hot_dataset
    




    