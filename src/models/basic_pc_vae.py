import torch
import torch.nn as nn
class vae_test(nn.Module):
    def __init__(self, max_seq_length, latent_dim):
        super().__init__()
        self.max_seq_length = max_seq_length
        self.latent_dim = latent_dim

        # Encoder Layers 
        self.conv1 = nn.Conv1d(in_channels = 4, out_channels = 128, kernel_size = 1, stride = 1)
        self.conv2 = nn.Conv1d(in_channels = 128, out_channels = 256, kernel_size = 1, stride = 1)
        self.max_pool = nn.MaxPool1d(kernel_size = max_seq_length, stride = 1)
        self.mean_fc = nn.Linear(256, latent_dim)
        self.std_fc = nn.Linear(256, latent_dim)

        # Decoder Layers
        self.dec_fc1 = nn.Linear(latent_dim, 128)
        self.dec_fc2 = nn.Linear(128, 256)
        self.dec_fc3 = nn.Linear(256, self.max_seq_length*4)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterization(mu, logvar)
        decoded = self.decoder(z)
        return decoded, mu, logvar
    
    def reparameterization(self, mean, var):
        epsilon = torch.rand_like(var)    
        z = mean + var*epsilon
        return z
    
    def encoder(self, x):
        x = x.permute(0,2,1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.max_pool(x)
  
        x = x.permute(0,2,1)
        return self.mean_fc(x), self.std_fc(x)
    
    def decoder(self, z):
        z = self.dec_fc1(z)
        z = self.dec_fc2(z)
        z = self.dec_fc3(z)
        z = self.sig(z)
        return z.view(z.shape[0],self.max_seq_length,4)