import torch
import torch.nn as nn

class mnist_vae(nn.Module):
    def __init__(self, latent_dim, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        # Activation Functions
        relu = nn.ReLU()
        sig = nn.Sigmoid()

        # Encoder 
        self.fc1_enc = nn.Linear(28*28, 256)
        self.fc2_enc = nn.Linear(256, 128)
        self.fc3_enc_mean = nn.Linear(128, latent_dim)
        self.fc3_enc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(latent_dim,128)
        self.fc2_dec = nn.Linear(128, 256)
        self.fc3_dec = nn.Linear(256, 28*28)

    def forward(self, x):
        reparam_z, x_mu, x_logvar = self.encode(x)
        x_rec = self.decode(reparam_z)
        return reparam_z, x_mu, x_logvar, x_rec

    def encode(self, x):
        x = self.relu(self.fc1_enc(x))
        x = self.relu(self.fc2_enc(x))

        x_mu = self.relu(self.fc3_enc_mean(x))
        x_logvar = self.relu(self.fc3_enc_logvar(x))

        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z, x_mu, x_logvar

    def decode(self, z):
        z = self.relu(self.fc1_dec(z))
        z = self.relu(self.fc2_dec(z))
        z = self.sig(self.fc3_dec(z))

        return z

    def reparametrisation(self, x_mu, x_logvar):
        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std).to(self.device)
        z_new = x_mu + eps*(std)
        return z_new

    def generate_n_samples(self, n):
        z = torch.randn(n, self.z_dim).to(self.device)
        return self.decode(z)

