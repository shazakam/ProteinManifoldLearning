import torch 
from torch.utils.data import Dataset
from src.utils.data_utils import *

class SequenceDataset(Dataset):
    """One Hot Encoded Sequence dataset either for Transformer or MLP model input."""

    def __init__(self, protein_shake_dataset, max_seq_len, transformer_input = False, return_proteins = True):
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
            self.enc_seq, self.org_protein_data = make_one_hot_data_list(self.protein_shake_dataset, 
                                                                         self.max_seq_len, 
                                                                         transformer_input = transformer_input, 
                                                                         return_proteins=return_proteins)
        else:
            self.enc_seq = make_one_hot_data_list(self.protein_shake_dataset, 
                                                  self.max_seq_len, 
                                                  transformer_input = transformer_input, 
                                                  return_proteins=return_proteins)
    def __len__(self):
        return len(self.enc_seq)

    def __getitem__(self, idx):
        sequence = self.enc_seq[idx]
        return sequence