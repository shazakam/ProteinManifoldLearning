import torch.nn as nn
import torch

class T_NET(nn.Module):
    
    def __init__(self, dim, embed_dim = 64, seq_len = 500):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(dim, embed_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(embed_dim, 2*embed_dim, kernel_size=1)
        self.conv3 = nn.Conv1d(2*embed_dim, embed_dim*8,kernel_size=1)

        self.fc1 = nn.Linear(embed_dim*8,2*embed_dim)
        self.fc2 = nn.Linear(2*embed_dim,embed_dim)
        self.fc3 = nn.Linear(embed_dim, dim**2)

        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.bn2 = nn.BatchNorm1d(embed_dim*2)
        self.bn3 = nn.BatchNorm1d(embed_dim*8)
        self.bn4 = nn.BatchNorm1d(embed_dim*2)
        self.bn5 = nn.BatchNorm1d(embed_dim)

        self.max_pool = nn.MaxPool1d(kernel_size = seq_len)
        self.relu = nn.ReLU()

        self.register_buffer('id', torch.eye(dim))


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))

        x = self.max_pool(x).view(batch_size, -1)

        x = self.bn4(self.relu(self.fc1(x)))
        x = self.bn5(self.relu(self.fc2(x)))
        x = self.fc3(x)

        id = self.id.repeat(batch_size,1,1)
        x = x.view(-1,self.dim, self.dim) + id

        return x

        