import torch.nn as nn
import torch 
import pytorch_lightning as pl
import torch_geometric
from torch_geometric.nn import GAE, VGAE, GCNConv, TopKPooling, global_mean_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

class GraphVAE(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, seq_len = 500, amino_acids = 20, conv_hidden_dim = 16, hidden_dim = 512, beta = 1, beta_increment = 0,  beta_epoch_start = 20, beta_cycle = 10, reconstruction_loss_weight = 1):

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
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.beta_increment = beta_increment
        self.beta_epoch_start = beta_epoch_start
        self.beta_cycle = beta_cycle

        # Encoder
        self.conv1 = GCNConv(self.amino_acids, self.conv_hidden_dim)
        self.conv2 = GCNConv(self.conv_hidden_dim, 2*self.conv_hidden_dim)

        self.fc_mu = nn.Linear(2*self.conv_hidden_dim*seq_len, self.latent_dim)
        self.fc_logvar = nn.Linear(2*self.conv_hidden_dim*seq_len, self.latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(self.latent_dim, hidden_dim)
        self.fc2_dec_feature = nn.Linear(hidden_dim, seq_len*self.amino_acids)

        self.fc_adj_dec = nn.Linear(self.latent_dim, 2*seq_len*conv_hidden_dim)

        # Activation Functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.soft = nn.Softmax(dim=-1)



    def forward(self, x):

        reparam_z, x_mu, x_logvar = self.encode(x)
        x_rec, logit_feature, adj_matrix = self.decode(reparam_z)

        return reparam_z, x_mu, x_logvar, x_rec, logit_feature, adj_matrix

    def encode(self, x):
        x_f = x.x
        x_edge_index = x.edge_index

        x = self.conv1(x_f, x_edge_index)
        x = self.tanh(x)
        x = self.conv2(x, x_edge_index)
        x = self.tanh(x)
        x = x.reshape(-1, self.seq_len*2*self.conv_hidden_dim)

        x_mu = self.fc_mu(x)
        x_logvar = self.fc_logvar(x)
        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z.squeeze(), x_mu.squeeze(), x_logvar.squeeze()

    def decode(self, z):

        adj_matrix = self.fc_adj_dec(z)
        adj_matrix = self.inner_product_decoder(adj_matrix)

        logit_feature_matrix = self.tanh(self.fc1_dec(z))
        logit_feature_matrix = self.fc2_dec_feature(logit_feature_matrix)
        logit_feature_matrix = logit_feature_matrix.reshape(-1,self.seq_len, self.amino_acids)

        z = self.soft(logit_feature_matrix)

        return z, logit_feature_matrix, adj_matrix

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new
    
    def inner_product_decoder(self, x):
        x = x.reshape(-1, self.seq_len, 2*self.conv_hidden_dim)
        x = self.sigmoid(torch.bmm(x, x.permute(0,2,1))) 
        return x
    
    def BCE_Loss(self, adj_matrix, x, x_true_indices):
        adj_targ = []
        for batch in range(x.batch[-1]+1):
            adj_targ.append(to_dense_adj(x[batch].edge_index, max_num_nodes=500).squeeze())
        adj_targ = torch.stack(adj_targ)

        valid_mask = (x_true_indices != -1).int() 
        mask_2d = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
        bce_loss = torch.nn.functional.binary_cross_entropy(torch.nn.functional.sigmoid(adj_matrix),
                                                            torch.nn.functional.sigmoid(adj_targ), reduction='none')

        return torch.sum((bce_loss*mask_2d)) / torch.sum(mask_2d)

    
    def ELBO(self, x, logit_feature, adj_matrix, x_mu, x_logvar):
        
        batch_size = x.batch[-1]+1
        x_true_feature = x.x.reshape(-1,self.seq_len, self.amino_acids)
        x_true_indices = x_true_feature.argmax(dim=-1)
        x_true_indices[torch.where(torch.sum(x_true_feature, dim = -1) == 0)] = -1

        feature_construction_loss = torch.nn.functional.cross_entropy(logit_feature.permute(0,2,1),x_true_indices, reduction='mean', ignore_index=-1)
        adjacency_construction_loss = self.BCE_Loss(adj_matrix, x, x_true_indices)

        rec_loss =  self.reconstruction_loss_weight*(feature_construction_loss+adjacency_construction_loss)
        KL_loss = self.beta*(-0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp()))

        return rec_loss + KL_loss/batch_size, rec_loss, adjacency_construction_loss, KL_loss/ batch_size
    
    
    def training_step(self, batch, batch_idx):

        x = batch
        batch_size = x.batch[-1]+1
        rep_z, x_mu, x_logvar, x_rec, logit_feature, adj_matrix = self(x)
        loss, rec_loss, adjacency_construction_loss, KL_loss = self.ELBO(x, logit_feature, adj_matrix, x_mu, x_logvar)
        self.log("train_adj loss", adjacency_construction_loss,on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("train_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.log("train_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
     
        return loss
    
    def validation_step(self, batch, batch_idx):

        x = batch
        batch_size = x.batch[-1]+1
        rep_z, x_mu, x_logvar, x_rec, logit_feature, adj_matrix = self(x)
        loss, rec_loss,adjacency_construction_loss, KL_loss = self.ELBO(x, logit_feature, adj_matrix, x_mu, x_logvar)
        
        self.log("val_adj loss", adjacency_construction_loss, on_step=True, on_epoch=True)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        self.log("val_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)
        self.log("val_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True, batch_size=batch_size)

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

        if self.current_epoch >= self.beta_epoch_start and (self.current_epoch - self.beta_epoch_start%self.beta_cycle) % self.beta_cycle == 0:
            self.beta += self.beta_increment  # Increase by a small amount
            self.log("beta", self.beta, prog_bar=True)  