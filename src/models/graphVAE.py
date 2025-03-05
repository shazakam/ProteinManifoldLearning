import torch.nn as nn
import torch 
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import GAE, VGAE, GCNConv, TopKPooling

class GraphVAE(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, k_pooling = 256, seq_len = 500, amino_acids = 20, conv_hidden_dim = 64, hidden_dim=512, dropout = 0.4, beta = 1):

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
        self.conv_hidden_dim = conv_hidden_dim
        self.k_pooling = k_pooling

        # Encoder
        self.conv1 = GCNConv(self.amino_acids, self.conv_hidden_dim)
        self.conv2 = GCNConv(self.conv_hidden_dim, 2*self.conv_hidden_dim)
        self.topk_pool = TopKPooling(2*self.conv_hidden_dim, ratio=int(self.k_pooling))
        self.fc_mu = nn.Linear(self.k_pooling*2*self.conv_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.k_pooling*2*self.conv_hidden_dim, self.latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(self.latent_dim, 2*self.conv_hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, self.input_dim)


    def forward(self, x):

        reparam_z, x_mu, x_logvar = self.encode(x)
        x_rec = self.decode(reparam_z)

        return reparam_z, x_mu, x_logvar, x_rec

    def encode(self, x):
        x_f = x.x
        x_edge_index = x.edge_index
        x_batch = x.batch

        x = self.conv1(x_f, x_edge_index)
        x = self.relu(x)
        x = self.conv2(x, x_edge_index)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.topk_pool(x, x_edge_index, None, x_batch)
        x = x.reshape(-1, self.k_pooling*2*self.conv_hidden_dim)
        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)

        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z.squeeze(), x_mu.squeeze(), x_logvar.squeeze()

    def decode(self, z):
        
        z = self.relu(self.fc1_dec(z))
        logit = self.fc3_dec(z)
        logit = logit.reshape(-1,self.seq_len, self.amino_acids)
        z = self.soft(logit)

        return z, logit

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new
    
    def ELBO(self, x, logit, x_mu, x_logvar):
        
        batch_size = x.batch[-1]
        indices_with_mask_val = []
        for i in range(batch_size):
            x_true_indices = x.x[torch.where(x.batch == i)[0]]
            x_true_indices = torch.nn.functional.pad(x_true_indices, (0,self.seq_len - x_true_indices.shape[0]), value=21)
            indices_with_mask_val.append(x_true_indices)
        indices_with_mask_val = torch.stack(indices_with_mask_val)

        rec_loss =  torch.nn.functional.cross_entropy(logit.permute(0,2,1),x_true_indices, reduction='sum', ignore_index=21)
        KL_loss = -0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp())

        return (rec_loss + self.beta*KL_loss) / x.size(0), rec_loss/ x.size(0), KL_loss/ x.size(0)
    
    
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