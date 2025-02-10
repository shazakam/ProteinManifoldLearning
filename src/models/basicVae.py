import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitBasicVae(pl.LightningModule):
    def __init__(self, latent_dim,  input_dimension, optimizer, optimizer_param, hidden_dim=512):
        """
        Initialize the BasicVae model.

        Args:
            latent_dim (int): Dimension of the latent space.
            input_dimension (int): Dimension of the input data.
            device (torch.device): Device to run the model on (e.g., 'cpu' or 'cuda').
            hidden_dim (int, optional): Dimension of the hidden layers. Default is 512.
        """
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.input_dim = input_dimension
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        # self.automatic_optimization = False

        # Activation Functions
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

        # Encoder 
        self.fc1_enc = nn.Linear(input_dimension, self.hidden_dim)
        self.fc3_enc_mean = nn.Linear(self.hidden_dim, latent_dim)
        self.fc3_enc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(latent_dim,self.hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, input_dimension)

    def forward(self, x):
        """
        Forward pass through the VAE.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Reparameterized latent vector, mean, log variance, and reconstructed input.
        """
        reparam_z, x_mu, x_logvar = self.encode(x)
        x_rec = self.decode(reparam_z)
        return reparam_z, x_mu, x_logvar, x_rec

    def encode(self, x):
        """
        Encode the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Reparameterized latent vector, mean, and log variance.
        """
        x = self.relu(self.fc1_enc(x))
        x_mu = self.fc3_enc_mean(x)
        x_logvar = self.fc3_enc_logvar(x)
        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z, x_mu, x_logvar

    def decode(self, z):
        """
        Decode the latent vector back to the input space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input.
        """
        z = self.relu(self.fc1_dec(z))
        z = self.sig(self.fc3_dec(z))

        return z

    def reparametrisation(self, x_mu, x_logvar):
        """
        Reparameterize the latent vector using the mean and log variance.

        Args:
            x_mu (torch.Tensor): Mean of the latent vector.
            x_logvar (torch.Tensor): Log variance of the latent vector.

        Returns:
            torch.Tensor: Reparameterized latent vector.
        """
        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) #.to(self.device)
        z_new = x_mu + eps*(std)

        return z_new

    def generate_n_samples(self, n):
        """
        Generate n samples from the latent space.

        Args:
            n (int): Number of samples to generate.

        Returns:
            torch.Tensor: Generated samples.
        """
        z = torch.randn(n, self.z_dim) #.to(self.device)
        return self.decode(z)
    
    def ELBO(self, x, x_hat,x_mu, x_logvar):
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
    
    def training_step(self, batch, batch_idx):
        """
        Training step for the VAE.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x = batch[0].view(-1, self.input_dim)
        rep_z, x_mu, x_logvar, x_rec = self(x)
        loss = self.ELBO(x, x_rec,x_mu, x_logvar)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        # opt = self.optimizers()
        # opt.zero_grad()
        # self.manual_backward(loss)
        # opt.step()
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the VAE.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Validation loss.
        """
        x = batch[0].view(-1, self.input_dim)
        rep_z, x_mu, x_logvar, x_rec = self(x)
        loss = self.ELBO(x, x_rec,x_mu, x_logvar)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        """
        Configure the optimizer for training.

        Returns:
            torch.optim.Optimizer: Optimizer.
        """
        return  self.optimizer(self.parameters(), **self.optimizer_param)

