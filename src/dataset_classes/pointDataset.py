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


    def __len__(self):
        return len(self.point_data)

    def __getitem__(self, idx):
        point_protein = self.point_data[idx]
        return point_protein
    