import torch 
import torch.nn as nn
import pytorch_lightning as pl
from ..utils.data_utils import *
import math

# TODO: CITE Relevant PyTorch Documentation for Attention Encoder Implementation

class AttentionVAE(pl.LightningModule):
    def __init__(self, optimizer, optimizer_param, N_layers, embed_dim, hidden_dim, num_heads, dropout, latent_dim, seq_len, get_attention_weights = False):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        self.N_layers = N_layers
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.get_attention_weights = False
        self.save_hyperparameters()

        # Encoder
        self.pos_encoder = PositionalEncoding(device=self.device, embed_dim=embed_dim,dropout=dropout)
        self.attention_encoders = nn.ModuleList([AttentionEncoderBlock(embed_dim=embed_dim,num_heads=num_heads,dropout=dropout, hidden_dim=hidden_dim) for i in range(N_layers)])

        self.global_forward_layer = nn.Linear(self.seq_len, self.seq_len//2)

        self.forward_latent_mean_layer = nn.Linear(self.embed_dim*self.seq_len//2,latent_dim)
        self.forward_latent_logvar_layer = nn.Linear(self.embed_dim*self.seq_len//2,latent_dim)
        self.soft = nn.Softmax(dim=-1)

        # Decoder
        self.relu = nn.ReLU()
        self.fc1_dec = nn.Linear(latent_dim,self.embed_dim*self.seq_len//2)
        self.fc3_dec = nn.Linear(self.embed_dim*self.seq_len//2, self.embed_dim*self.seq_len)

        self.attention_weights = [0]*N_layers



    def forward(self, x):
            """
            Forward pass through the AttentionVAE.

            Args:
                x (torch.Tensor): Input tensor.

            Returns:
                tuple: Reparameterized latent vector, mean, log variance, and reconstructed input.
            """

            reparam_z, x_mu, x_logvar = self.encode(x)
            x_rec, logit = self.decode(reparam_z)
            return reparam_z, x_mu, x_logvar, x_rec, logit

    def encode(self, x):
        """
        Encode the input into the latent space.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            tuple: Reparameterized latent vector, mean, and log variance.
        
        """

        # Create Mask to ignore empty tokens
        padding_mask = (x.sum(dim=-1) == 0).bool()

        # Add Positional Encoding information  
        x = self.pos_encoder(x)

        # Perform Multihead self attention to get a context aware embedding of sequence input
        for idx, attention_enc in enumerate(self.attention_encoders):

            if self.get_attention_weights:
                x, attention_weight = attention_enc(x, key_mask = padding_mask)
                self.attention_weights[idx] = attention_weight
            else:
                x, _ = attention_enc(x, key_mask = padding_mask)

        # Permute and convert sequence to a shortened representation (N, embed_dim, shortened seq_len)
        x = x.permute(0,2,1)
        x = self.global_forward_layer(x)
        
        # Flatten Final representation
        x = x.reshape(-1, x.shape[1]*x.shape[2])

        # Pass shortened and context aware embedding through final layers for latent representation
        x_mu = self.forward_latent_mean_layer(x)
        x_logvar = self.forward_latent_logvar_layer(x)
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
        logit = self.fc3_dec(z)
        logit = logit.reshape(-1,self.seq_len, self.embed_dim)
        z = self.soft(logit)
        return z, logit
    

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

    def ELBO(self, x, logit, x_mu, x_logvar):
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
        x_true_indices = x.argmax(dim=-1)
        rec_loss =  torch.nn.functional.cross_entropy(logit.permute(0,2,1),x_true_indices, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())
        self.log("av_rec_loss", rec_loss / x.size(0), prog_bar=True)
        self.log("av_KL_loss", KL_loss / x.size(0), prog_bar=True)
        return ((rec_loss) + (KL_loss)) / x.size(0)

    def training_step(self, batch, batch_idx):
        """
        Training step for the VAE.

        Args:
            batch (tuple): Batch of data.
            batch_idx (int): Batch index.

        Returns:
            torch.Tensor: Training loss.
        """
        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss = self.ELBO(x, logit, x_mu, x_logvar)
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
        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss = self.ELBO(x, logit ,x_mu, x_logvar)
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
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train_loss_epoch",  # Reduce LR based on training loss
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  
            }
        }
    


    def on_train_start(self):
        self.pos_encoder.device = self.device

    def on_train_epoch_end(self):
        return super().validation_step()
    
    def on_train_epoch_end(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)
    
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
    
    def forward(self, x, key_mask):

        # Multihead Attention and Add+Layer Norm 
        attention_out, attention_weights = self.attention(x, x, x, key_padding_mask = key_mask, need_weights = True)
        x = x + self.dropout(attention_out)
        x = self.layer_norm(x)

        # FeedForward
        x = self.feedforward_sublayer(x)
        return x, attention_weights
    
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

    def __init__(self, device, embed_dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.device = device
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
        y = self.pe[:x.size(0)].to(self.device)
        x = x + y
     
        return self.dropout(x)