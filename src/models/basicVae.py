import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitBasicVae(pl.LightningModule):
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
        self.input_dim = (self.amino_acids)*self.seq_len

        self.log("beta", beta, logger=True)

        # Activation Functions
        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=2)
        self.dropout_layer = nn.Dropout(dropout)

        # Encoder 
        # self.conv1 = nn.Conv1d(in_channels = self.amino_acids, out_channels = 2*self.amino_acids, kernel_size = 1)
        # self.conv2 = nn.Conv1d(in_channels = 2*self.amino_acids, out_channels = 2*self.amino_acids, kernel_size = 5, padding='same')
        self.fc1_enc = nn.Linear(self.amino_acids*self.seq_len, self.hidden_dim)
        self.fc3_enc_mean = nn.Linear(self.hidden_dim, latent_dim)
        self.fc3_enc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(latent_dim,self.hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):

        # x = x.view(-1,self.input_dim)
        # X has shape (B, S, A)
        reparam_z, x_mu, x_logvar = self.encode(x)
        # Reparam, x_mu and x_logvar have shape (B, latent_dim)
        x_rec, logit = self.decode(reparam_z)
        # x_rec and x_logit have shape (B,S,A)
        return reparam_z, x_mu, x_logvar, x_rec, logit

    def encode(self, x):
        # x = self.conv1(x.permute(0,2,1))
        # x = self.relu(x)
        # x = self.conv2(x)
        x = x.reshape(-1,(self.amino_acids)*self.seq_len)

        x = self.relu(self.fc1_enc(x))
        x_mu = self.relu(self.fc3_enc_mean(x))
        x_logvar = self.relu(self.fc3_enc_logvar(x))
        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z, x_mu, x_logvar

    def decode(self, z):

        z = self.dropout_layer(self.relu(self.fc1_dec(z)))
        logit = self.dropout_layer(self.fc3_dec(z))
        logit = logit.reshape(-1,self.seq_len, self.amino_acids)
        z = self.soft(logit)

        return z, logit

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new
    
    def ELBO(self, x, logit, x_mu, x_logvar):
        # X and logit have shape (B, S, A)
        # x_mu and x_logvar have shape (B, latent_dim)
        
        # X true indices contains index locations of true labels
        x_true_indices = x.argmax(dim=-1)

        # Permute logit to shape (B, A, S) for cross entropy function and ignore index 21 (padding value)
        rec_loss =  torch.nn.functional.cross_entropy(logit.permute(0,2,1),x_true_indices, reduction='sum', ignore_index=20)
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())
        
        return (rec_loss + self.beta*KL_loss) / x.size(0), rec_loss/ x.size(0), KL_loss/ x.size(0)
    
    def training_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(x, logit, x_mu, x_logvar)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
     
        return loss
    
    def validation_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(x, logit,x_mu, x_logvar)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)

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

        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)

