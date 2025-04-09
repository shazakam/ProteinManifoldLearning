import torch 
from torch.utils.data import Dataset
from src.utils.data_utils import *

class PointDataset(Dataset):

    def __init__(self, protein_shake_dataset, max_seq_len, return_proteins = False):
        """
        Arguments:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.protein_shake_dataset = protein_shake_dataset
        self.max_seq_len = max_seq_len
        if return_proteins:
            self.point_data, self.org_protein_data = filter_by_max_length_and_pad(protein_shake_dataset,max_seq_len, return_proteins)
        else:
            self.point_data = filter_by_max_length_and_pad(protein_shake_dataset,max_seq_len, return_proteins)

        max_dist = self.find_max_dist_from_zero()
        self.point_data = torch.stack(self.point_data)
        normalised_dist = self.point_data[:,:,:3] / max_dist
        self.point_data[:,:,:3] = normalised_dist

    def __len__(self):
        return len(self.point_data)
    
    def find_max_dist_from_zero(self):
        max_dist = 0
        for sample in self.point_data:
            max_dist_in_sample = torch.max(torch.abs(torch.sqrt(torch.sum(sample[:,:3]**2, axis = 1))))
            if max_dist < max_dist_in_sample:
                max_dist = max_dist_in_sample
        return max_dist

    def __getitem__(self, idx):
        point_protein = self.point_data[idx]
        return point_protein
    