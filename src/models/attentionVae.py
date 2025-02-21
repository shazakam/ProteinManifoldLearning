import torch 
import torch.nn as nn
import pytorch_lightning as pl
from ..utils.data_utils import *
import math

# TODO: CITE Relevant PyTorch Documentation for Attention Encoder Implementation

class AttentionVAE(pl.LightningModule):
    def __init__(self, optimizer, optimizer_param, embed_dim, hidden_dim, num_heads, dropout, latent_dim, seq_len, device = 'mps'):
        super().__init__()
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.num_unique_tokens = 21
        self.save_hyperparameters()

        # Encoder
        self.dropout_layer = nn.Dropout(0.3)
        self.embedding_layer = nn.Embedding(num_embeddings = self.num_unique_tokens, embedding_dim = embed_dim, padding_idx = 20)

        position = torch.arange(seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        self.pe = torch.zeros(seq_len, embed_dim)
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)
        self.pe = self.pe.to(device)

        # self.pos_encoder = PositionalEncoding(device=self.device, embed_dim=embed_dim,dropout=dropout)

        self.transformer_encoder = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)

        self.global_forward_layer = nn.Linear(self.seq_len, self.hidden_dim)

        self.forward_latent_mean_layer = nn.Linear(self.embed_dim*self.hidden_dim, latent_dim)
        self.forward_latent_logvar_layer = nn.Linear(self.embed_dim*self.hidden_dim, latent_dim)
        self.soft = nn.Softmax(dim=-1)

        # Decoder
        self.relu = nn.ReLU()
        self.fc1_dec = nn.Linear(latent_dim, self.num_unique_tokens*self.seq_len)
        # self.fc3_dec = nn.Linear(self.num_unique_tokens*(self.seq_len//2), self.num_unique_tokens*self.seq_len)

        self.batch_KL_loss = 0
        self.batch_rec_loss = 0



    def forward(self, x):
            # Embed protein sequence
            x = self.embedding_layer(x)

            reparam_z, x_mu, x_logvar = self.encode(x)
            x_rec, logit = self.decode(reparam_z)
            return reparam_z, x_mu, x_logvar, x_rec, logit

    def encode(self, x):
        # Create Mask to ignore empty tokens
        padding_mask = (x.sum(dim=-1) == 0).bool()

        # Add Positional Encoding information  
        # x = self.pos_encoder(x)

        x = x + self.pe[:, :x.size(1)]
        x = self.dropout_layer(x)

        x = self.transformer_encoder(src = x, src_key_padding_mask = padding_mask)

        # Permute and convert sequence to a shortened representation (N, embed_dim, shortened seq_len)
        x = x.permute(0,2,1)
        x = self.global_forward_layer(x)
        x = self.relu(x)
        
        # Flatten Final representation
        x = x.reshape(-1, x.shape[1]*x.shape[2])

        # Pass shortened and context aware embedding through final layers for latent representation
        x_mu = self.forward_latent_mean_layer(x)
        x_logvar = self.forward_latent_logvar_layer(x)
        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z, x_mu, x_logvar
    
    def decode(self, z):

        z = self.relu(self.dropout_layer(self.fc1_dec(z)))
        # NOTE Removed fc3 here 
        logit = z
        logit = logit.reshape(-1,self.seq_len, self.num_unique_tokens)
        z = self.soft(logit)
        return z, logit
    

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std)
        z_new = x_mu + eps*(std)

        return z_new

    def ELBO(self, x, logit, x_mu, x_logvar):
  
        x_true_indices = x.argmax(dim=-1)
        rec_loss =  torch.nn.functional.cross_entropy(logit.permute(0,2,1),x_true_indices, reduction='sum')
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())
    
        return ((rec_loss) + (KL_loss)) / x.size(0), rec_loss/ x.size(0), KL_loss/ x.size(0)

    def training_step(self, batch, batch_idx):

        x, one_hot_encoded_x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(one_hot_encoded_x, logit, x_mu, x_logvar)

        self.batch_rec_loss += rec_loss.item()
        self.batch_KL_loss += KL_loss.item()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("average_train_epoch_rec_loss", self.batch_rec_loss / x.shape[0], on_epoch=True, on_step=False, prog_bar=True)
        self.log("average_train_epoch_KL_loss", self.batch_KL_loss / x.shape[0], on_epoch=True, on_step=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):

        x, one_hot_encoded_x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(one_hot_encoded_x, logit ,x_mu, x_logvar)

        self.batch_rec_loss += rec_loss.item()
        self.batch_KL_loss += KL_loss.item()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("average_val_epoch_rec_loss", self.batch_rec_loss / x.shape[0], on_epoch=True, on_step=False, prog_bar=True)
        self.log("average_val_epoch_KL_loss", self.batch_rec_loss / x.shape[0], on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def configure_optimizers(self):

        optimizer = self.optimizer(self.parameters(), **self.optimizer_param)

        # 🔹 Using ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)


        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Reduce LR based on training loss
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  
            }
        }

    # def on_train_start(self):
    #     self.pos_encoder.device = self.device

    # def on_validation_start(self):
    #     self.pos_encoder.device = self.device
    
    def on_train_epoch_end(self):
        self.batch_rec_loss = 0
        self.batch_KL_loss = 0

        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)

    def on_validation_epoch_end(self):
        self.batch_rec_loss = 0
        self.batch_KL_loss = 0
    
class PositionalEncoding(nn.Module):

    def __init__(self, device, embed_dim: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.device = device
        # self.dropout = nn.Dropout(p=dropout)

        # position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))

        # self.pe = torch.zeros(max_len, embed_dim)
        # self.pe[:, 0::2] = torch.sin(position * div_term)
        # self.pe[:, 1::2] = torch.cos(position * div_term)
        # self.pe = self.pe.unsqueeze(0)

    def forward(self, x):

        y = self.pe[:, :x.size(1)].to(self.device).requires_grad_(False)
        x = x + y

        del y
     
        return self.dropout(x)