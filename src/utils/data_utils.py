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

def one_hot_encode_seq(seq, max_seq_len, transformer_input = False, convert_to_tensor=True):

    amino_encoding_dict = {'S': 0,'Y': 1,'F': 2,'K': 3,'V': 4,'C': 5,
                            'A': 6,'H': 7,'T': 8,'I': 9,'W': 10,'R': 11,
                            'P': 12,'L': 13,'G': 14,'N': 15,'E': 16,
                            'Q': 17,'M': 18,'D': 19, 'X':20}
    
    one_hot_list = []

    if not transformer_input:
        seq = seq+(max_seq_len-len(seq))*'X'

        for char in seq:
            one_hot = np.zeros(21)
            one_hot[amino_encoding_dict[char]] = 1
            one_hot_list.append(one_hot)
        return torch.tensor(np.array(one_hot_list),dtype=torch.float32)
    else:

        for char in seq:
            one_hot = np.zeros(20)
            one_hot[amino_encoding_dict[char]] = 1
            one_hot_list.append(one_hot)

        # Padd Sequence with zeros if not long enough
        encoded_array = np.array(one_hot_list)
        padded_array = np.pad(encoded_array, pad_width=((0, max_seq_len - encoded_array.shape[0]), (0, 0)), mode='constant', constant_values=0)

        # # Check to see if the input is to be used in a transformer encoder
        # if not transformer_input:

        #     # Flatten array for MLP input, convert to tensor #TODO: Just get rid of the np implementation at some point
        #     if convert_to_tensor:
        #         return torch.tensor(padded_array).float().flatten().view(1,-1)
        #     else:
        #         return padded_array.flatten().reshape(1,-1)
            
        # else:
        return torch.tensor(padded_array, dtype=torch.float32)
    

def make_one_hot_data_list(dataset, max_seq_len, transformer_input):
    one_hot_dataset = []
    proteins = []

    for sample in tqdm(dataset):

        if len(sample[1]['protein']['sequence']) > max_seq_len:
            continue

        else:
            proteins.append(sample)
            one_hot_seq = one_hot_encode_seq(sample[1]['protein']['sequence'], max_seq_len, transformer_input)
            # one_hot_seq = one_hot_seq.flatten()
            one_hot_dataset.append(one_hot_seq)

    return one_hot_dataset, proteins

def ELBO(x, x_hat,x_mu, x_logvar):
        """
        Compute the Evidence Lower Bound (ELBO) loss.

        Args:
            x (torch.Tensor): Original input.
            x_hat (torch.Tensor): Reconstructed input.
            x_mu (torch.Tensor): Mean of the latent vector.
            x_logvar (torch.Tensor): Log variance of the latent vector.

        Returns:
            torch.Tensor: ELBO loss.
        """
        rec_loss =  torch.nn.functional.mse_loss(x_hat, x, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())

        return (rec_loss + KL_loss) / x.size(0) 


    