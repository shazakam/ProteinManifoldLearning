import torch
import torch_geometric
from proteinshake.datasets import ProteinFamilyDataset
from torch_geometric.transforms import Pad
from torch_geometric.data import Dataset

# One hot encodes graph nodes in dataset
def load_graph_data(dataset, amnino_acids = 21, max_seq_len = 500):
    pad_transform = Pad(max_num_nodes = max_seq_len, add_pad_mask=True, max_num_edges=9010)
    dataset = [data[0] for data in dataset if data[0].x.shape[0] < max_seq_len]
    for idx, data in enumerate(dataset):
        data.x = torch.cat([data.x, torch.tensor([20])])
        data.x = torch.nn.functional.one_hot(data.x, num_classes=amnino_acids).float()

        dataset[idx] = pad_transform(data)

    return dataset


class GraphListDataset(Dataset):
    def __init__(self, graph_list):
        super(GraphListDataset, self).__init__()
        self.graph_list = graph_list
    
    def len(self):
        return len(self.graph_list)
    
    def get(self, idx):
        return self.graph_list[idx]