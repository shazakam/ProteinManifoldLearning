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

    if return_proteins:
        org_protein_data = []
        for sample in tqdm(dataset):
            seq_leng = len(sample[1]['protein']['sequence'])
            if seq_leng <= max_seq_len:
                proteins.append(pad_cloud_data(sample[0], max_seq_len))
                org_protein_data.append(sample)
        return proteins, org_protein_data
    
    else:
        for sample in tqdm(dataset):
            seq_leng = len(sample[1]['protein']['sequence'])
            if seq_leng <= max_seq_len:
                proteins.append(pad_cloud_data(sample[0], max_seq_len))
        return proteins
    
def pad_cloud_data(sample, max_seq_len):
    num_pads = max_seq_len - sample.shape[0]
    padding = torch.zeros((num_pads, sample.shape[1]))
    padded = torch.concatenate([sample, padding], dim = 0)
    return padded




def one_hot_encode_seq(seq, max_seq_len, transformer_input = False, convert_to_tensor=True):

    amino_encoding_dict = {'S': 0,'Y': 1,'F': 2,'K': 3,'V': 4,'C': 5,
                            'A': 6,'H': 7,'T': 8,'I': 9,'W': 10,'R': 11,
                            'P': 12,'L': 13,'G': 14,'N': 15,'E': 16,
                            'Q': 17,'M': 18,'D': 19, 'X':20}
    
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
    




    