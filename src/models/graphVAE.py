import torch.nn as nn
import torch 
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import GAE, VGAE, GCNConv

class GraphVAE(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, seq_len = 500, amino_acids = 21, hidden_dim=512, dropout = 0.4, beta = 1):

        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        self.seq_len = seq_len
        self.amino_acids = amino_acids
        self.input_dim = self.amino_acids*self.seq_len

        self.conv_mu = GCNConv(3, latent_dim)
        self.conv_logstd = GCNConv(3, latent_dim)


    def forward(self, x):

        reparam_z, x_mu, x_logvar = self.encode(x)
        x_rec = self.decode(reparam_z)

        return reparam_z, x_mu, x_logvar, x_rec

    def encode(self, x):
        x_mu = self.conv_mu(x)
        x_logvar = self.conv_logstd(x)


        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z.squeeze(), x_mu.squeeze(), x_logvar.squeeze()

    def decode(self, z):
        

        return z

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new
    
    def ELBO(self, x, x_reconstructed, x_mu, x_logvar):

        rec_loss =  self.chamfer_distance(x, x_reconstructed)
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())

        return (rec_loss + self.beta*KL_loss) / x.size(0), rec_loss/ x.size(0), KL_loss/ x.size(0)
    
    # def chamfer_distance(self,x,x_reconstructed):
    #     pairwise_dist = torch.cdist(x, x_reconstructed, p = 2)
    #     loss = torch.sum(torch.min(pairwise_dist, dim= -1)[0], dim = -1)
    #     return loss
    
    def training_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(x, logit, x_mu, x_logvar)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("average_train_epoch_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("average_train_epoch_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
     
        return loss
    
    def validation_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(x, logit,x_mu, x_logvar)

        self.batch_rec_loss.append(rec_loss)
        self.batch_KL_loss.append(KL_loss)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("average_val_epoch_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("average_val_epoch_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)

        return loss
    
    def configure_optimizers(self):

        optimizer = self.optimizer(self.parameters(), **self.optimizer_param)

        # 🔹 Using ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # Reduce LR based on validation loss
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  
            }
        }
    
    def on_train_epoch_end(self):
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)

        self.batch_rec_loss = []
        self.batch_KL_loss = []

        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)