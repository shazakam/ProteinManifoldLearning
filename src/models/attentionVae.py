import torch 
import torch.nn as nn
import pytorch_lightning as pl
from ..utils.data_utils import *
import math

# TODO: CITE Relevant PyTorch Documentation for Attention Encoder Implementation

class AttentionVAE(pl.LightningModule):
    def __init__(self, N_layers, embed_dim, hidden_dim, num_heads, dropout, latent_dim):
        super().__init__()
        self.N_layers = N_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.latent_dim = latent_dim

        # Encoders
        self.pos_encoder = PositionalEncoding(embed_dim=embed_dim,dropout=dropout)
        self.attention_encoders = nn.ModuleList([AttentionEncoderBlock(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout, hidden_dim=hidden_dim) for i in range(N_layers)])
        self.forward_latent_layer = nn.Linear(embed_dim,latent_dim)

        # Decoder



    def forward(self, x):
            """
            Forward pass through the AttentionVAE.

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
        return 
    def decode(self, z):
        """
        Decode the latent vector back to the input space.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed input.
        """

        return 

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
        eps = torch.randn_like(std)
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
        z = torch.randn(n, self.z_dim)
        return self.decode(z)
    
    # TODO: Add ELBO to data_utils for ease of use
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
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
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

        optimizer = self.optimizer(self.parameters(), **self.optimizer_param)

        # 🔹 Using ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)


        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",  # Reduce LR based on training loss
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  
            }
        }

    def on_train_epoch_end(self):
        return super().validation_step()
    
class AttentionEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        # Self Attention over Sequence inputs
        self.attention = nn.MultiheadAttention(embed_dim, num_heads,dropout, batch_first = True)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Fully Connected Feed Forward layers and Layer Norms as described by "Attention is All You need"
        self.feedforward_sublayer = FeedForwardEncoderSubLayer(embed_dim,hidden_dim, dropout)
    
    def forward(self, x):

        # Ignore Padded Tokens
        key_mask = (x.sum(dim=-1) == 0).bool()  

        # Multihead Attention and Add+Layer Norm 
        x , _ = x + self.dropout(self.attention(x, x, x, key_padding_mask = key_mask))
        x = self.layer_norm(x)

        # FeedForward
        x = self.feedforward_sublayer(x)
        return x
    
class FeedForwardEncoderSubLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feedforward_out = self.feedforward(x)
        x = self.layer_norm(x+feedforward_out)
        return x
    
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        self.pe = torch.zeros(max_len, 1, embed_dim)
        self.pe[:, 0, 0::2] = torch.sin(position * div_term)
        self.pe[:, 0, 1::2] = torch.cos(position * div_term)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)