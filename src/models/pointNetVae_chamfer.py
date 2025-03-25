import torch
import torch.nn as nn
import pytorch_lightning as pl

class PointNetVAE(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, seq_len = 500, amino_acids = 21, hidden_dim=512, beta = 0, beta_increment = 1, beta_epoch_start = 20, beta_cycle = 20, conv_hidden_dim = 128, global_feature_size = 512, reconstruction_loss_weight = 1):

        super().__init__()
        self.save_hyperparameters()
        
        self.beta = beta
        self.beta_epoch_start = beta_epoch_start
        self.beta_increment = beta_increment
        self.beta_cycle = beta_cycle

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        self.seq_len = seq_len
        self.amino_acids = amino_acids
        self.input_dim = (self.amino_acids+3)*self.seq_len
        self.global_feature_size = global_feature_size
        self.reconstruction_loss_weight = reconstruction_loss_weight


        # Sequence Encoding Layer

        self.fc1_enc = nn.Linear(self.amino_acids*self.seq_len, self.global_feature_size)
        self.bn_label = nn.BatchNorm1d(self.global_feature_size) 

        # Final Linear Output Layers
        self.fc1_enc_mu = nn.Linear(2*self.global_feature_size, self.latent_dim)
        self.fc1_enc_logvar = nn.Linear(2*self.global_feature_size, self.latent_dim)

        self.conv1 = torch.nn.Conv1d(3, conv_hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(conv_hidden_dim, conv_hidden_dim*2, 1)
        self.conv3 = torch.nn.Conv1d(conv_hidden_dim*2, self.global_feature_size, 1)
        self.bn1 = nn.BatchNorm1d(conv_hidden_dim)
        self.bn2 = nn.BatchNorm1d(conv_hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(self.global_feature_size)

        self.tanh = nn.Tanh()
        self.soft = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool1d(kernel_size = seq_len)

        # Decoder
        self.fc1_dec = nn.Linear(latent_dim,self.hidden_dim)
        self.bn1_dec = nn.BatchNorm1d(hidden_dim) 
        self.fc3_dec = nn.Linear(self.hidden_dim, self.input_dim)
        self.bn3_dec = nn.BatchNorm1d(self.input_dim)


    def forward(self, x):
        labels = x[:,:,3:].float()
        x = x[:, :, :3]
        reparam_z, x_mu, x_logvar = self.encode(x, labels)
        x_rec, logit = self.decode(reparam_z)

        return reparam_z, x_mu, x_logvar, x_rec, logit

    def encode(self, x, labels):
        # X has shape (batch, seq_len, 3)
     
        # Encode Sequences
        # print(labels.shape)
        labels = labels.reshape(-1,self.amino_acids*self.seq_len)
        labels = self.tanh(self.bn_label(self.fc1_enc(labels)))

        x = self.tanh(self.bn1(self.conv1(x.permute(0,2,1))))
        x = self.tanh(self.bn2(self.conv2(x)))
        x = self.tanh(self.bn3(self.conv3(x)))

        global_features = self.max_pool(x).squeeze()
        global_features = torch.cat((global_features, labels), dim = -1)

        x_mu = self.fc1_enc_mu(global_features)
        x_logvar = self.fc1_enc_logvar(global_features)

        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z.squeeze(), x_mu.squeeze(), x_logvar.squeeze()

    def decode(self, z):
        
        z = self.tanh(self.bn1_dec(self.fc1_dec(z)))
        logit = self.fc3_dec(z)
        logit = logit.reshape(-1,self.seq_len, self.amino_acids+3)

        softmax_logits = self.soft(logit[:,:, 3:])
        z = torch.cat((logit[:, :, :3], softmax_logits), dim=-1)  
        return z, logit

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new

    def calculate_chamfer_distance(self, predicted_coords, true_coords, mask):

        AB_pwd = torch.cdist(predicted_coords, x2=true_coords, p=2)
        mask = (mask.unsqueeze(-1).float() @ mask.unsqueeze(1).float()).int().bool()
        masked_dist_matrix = torch.where(mask, AB_pwd, torch.tensor(float('inf'), device=mask.device)) 

        non_inf_sum = torch.sum(torch.where(torch.min(masked_dist_matrix, dim=-1)[0] != torch.inf, 
                                           torch.min(masked_dist_matrix, dim=-1)[0], 
                                           torch.tensor(0)), dim = -1)
        

        return 2*torch.mean(non_inf_sum)

    def calculate_mse(self, predicted_coords, true_coords, mask):

        mse = torch.sqrt(torch.sum(((true_coords-predicted_coords)**2), dim=-1))*mask

        return torch.sum(mse) / torch.sum(mask)
       
    def ELBO(self, x, logit, x_mu, x_logvar):
        x_true_indices = x[:,:,3:]
        x_true_indices = x_true_indices.argmax(dim=-1)
        x_true_indices[torch.where(torch.sum(x[:,:,3:], dim = -1) == 0)] = -1


        mask = (x_true_indices != -1)
        predicted_coords = logit[:,:,:3]
        logit = logit[:,:,3:]

        chamfer = self.calculate_chamfer_distance(predicted_coords, x[:,:,:3], mask)
        cross_entropy = torch.nn.functional.cross_entropy(logit.permute(0,2,1), x_true_indices, reduction='mean', ignore_index=-1)
        rec_loss = self.reconstruction_loss_weight*(cross_entropy + chamfer)

        KL_loss =self.beta*(-0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp()))
        return (rec_loss + KL_loss), rec_loss, KL_loss, chamfer
    
    
    def training_step(self, batch, batch_idx):

        x = batch[0]
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        
        loss, rec_loss, KL_loss, mse = self.ELBO(x, logit, x_mu, x_logvar)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rec_loss", rec_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("train_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("train_mse", mse, on_epoch=True, on_step=False, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):

        x = batch[0]
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss, mse = self.ELBO(x, logit, x_mu, x_logvar)
        # transform_loss = self.orthogonal_transform_regulariser(feature_transform, input_transform)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_mse", mse, on_epoch=True, on_step=False, prog_bar=True)
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