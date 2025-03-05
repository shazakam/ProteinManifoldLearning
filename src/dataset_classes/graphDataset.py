import torch
import torch_geometric
from proteinshake.datasets import ProteinFamilyDataset
# Define a function to load the actual dataset and one-hot encode features

# One hot encodes graph nodes in dataset
def load_graph_data(dataset, amnino_acids = 21, max_seq_len = 500):
    dataset = [data[0] for data in dataset]
    for data in dataset:
        data.x = torch.nn.functional.one_hot(data.x, num_classes=amnino_acids).float()

    return dataset