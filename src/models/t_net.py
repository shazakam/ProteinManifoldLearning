import torch.nn as nn
import torch

class T_NET(nn.Module):
    
    def __init__(self, dim, seq_len = 500):
        super().__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(dim, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64,128, kernel_size=1)
        self.conv3 = nn.Conv1d(128,1024,kernel_size=1)

        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,dim**2)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.max_pool = nn.MaxPool1d(kernel_size = seq_len)
        self.relu = nn.ReLU()


    def forward(self, x):
        batch_size = x.shape[0]
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.conv2(x)))
        x = self.bn3(self.relu(self.conv3(x)))

        x = self.max_pool(x).view(batch_size, -1)

        x = self.bn4(self.relu(self.fc1(x)))
        x = self.bn5(self.relu(self.fc2(x)))
        x = self.fc3(x)

        id = torch.eye(self.dim, requires_grad=True).repeat(batch_size,  1, 1)

        id.to(x.device)

        x = x.view(-1,self.dim, self.dim) + id

        return x

        