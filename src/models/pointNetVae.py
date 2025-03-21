import torch
import torch.nn as nn
import pytorch_lightning as pl

from .t_net import T_NET

class PointNetVAE(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, seq_len = 500, amino_acids = 21, hidden_dim=512, dropout = 0.4, beta = 1, conv_hidden_dim = 128, global_feature_size = 512, seq_embedding = 32, reconstruction_loss_weight = 0.1):

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
        self.global_feature_size = global_feature_size
        self.seq_embedding = seq_embedding
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # self.input_t_net = T_NET(dim = 3, seq_len = seq_len)
        # self.feature_t_net = T_NET(dim = embed_dim, seq_len = seq_len, embed_dim = embed_dim)

        self.embedding = nn.Embedding(21,seq_embedding,20)

        # Sequence Encoding Layer
        self.conv1_label = nn.Conv1d(in_channels = seq_embedding, out_channels = seq_embedding, kernel_size = 1)
        self.conv2_label = nn.Conv1d(in_channels = seq_embedding, out_channels = seq_embedding, kernel_size = 1)
        self.fc1_seq_enc = nn.Linear(seq_embedding*seq_len, self.global_feature_size)

        # # FIRST SHARED LAYER
        # self.shared_conv1d_1 = nn.Conv1d(in_channels = 3, out_channels = embed_dim, kernel_size = 1)
        # self.shared_conv1d_2 = nn.Conv1d(in_channels = embed_dim, out_channels = embed_dim, kernel_size = 1)


        # # SECOND SHARED LAYER
        # self.shared_conv1d_3 = nn.Conv1d(in_channels = embed_dim, out_channels = embed_dim, kernel_size = 1)
        # self.shared_conv1d_4 = nn.Conv1d(in_channels = embed_dim, out_channels = 128, kernel_size = 1)
        # self.shared_conv1d_5 = nn.Conv1d(in_channels = 128, out_channels = self.global_feature_size, kernel_size = 1)
        # # self.shared_conv1d_6 = nn.Conv1d(in_channels = 128, out_channels = self.latent_dim, kernel_size = 1)

        # self.bn1 = nn.BatchNorm1d(embed_dim)
        # self.bn2 = nn.BatchNorm1d(embed_dim)
        # self.bn3 = nn.BatchNorm1d(embed_dim)
        # self.bn4 = nn.BatchNorm1d(128)
        # self.bn5 = nn.BatchNorm1d(self.global_feature_size)
        # # self.bn6 = nn.BatchNorm1d(self.latent_dim)

        # Final Linear Output Layers
        self.fc1_enc_mu = nn.Linear(2*self.global_feature_size, self.latent_dim)
        self.fc1_enc_logvar = nn.Linear(2*self.global_feature_size, self.latent_dim)

        self.conv1 = torch.nn.Conv1d(3, conv_hidden_dim, 1)
        self.conv2 = torch.nn.Conv1d(conv_hidden_dim, conv_hidden_dim*2, 1)
        self.conv3 = torch.nn.Conv1d(conv_hidden_dim*2, self.global_feature_size, 1)
        self.bn1 = nn.BatchNorm1d(conv_hidden_dim)
        self.bn2 = nn.BatchNorm1d(conv_hidden_dim*2)
        self.bn3 = nn.BatchNorm1d(self.global_feature_size)


        self.relu = nn.ReLU()
        self.soft = nn.Softmax(dim=-1)
        self.max_pool = nn.MaxPool1d(kernel_size = seq_len)

        # DECODER Layers 
        self.fc1_dec = nn.Linear(self.latent_dim, self.latent_dim*2)
        self.fc2_dec = nn.Linear(self.latent_dim*2, self.latent_dim*4)
        self.fc3_dec = nn.Linear(self.latent_dim*4, self.latent_dim*8)
        self.fc4_dec = nn.Linear(self.latent_dim*8, self.seq_len*self.amino_acids)

        # self.register_buffer("ident1", torch.eye(3).view(1, 3, 3))
        # self.register_buffer("ident2", torch.eye(embed_dim).view(1, embed_dim, embed_dim))
     


    def forward(self, x):
        labels = x[:,:,3].long()
        x = x[:, :, :3]
        labels = self.embedding(labels)
        reparam_z, x_mu, x_logvar = self.encode(x, labels)
        x_rec, logit = self.decode(reparam_z)

        return reparam_z, x_mu, x_logvar, x_rec, logit

    def encode(self, x, labels):
        # X has shape (batch, seq_len, 3)
     
        # Encode Sequences
        labels = self.conv1_label(labels.permute(0,2,1))
        labels = self.relu(labels)
        labels = self.conv2_label(labels)
        labels = self.relu(labels)
        labels = labels.reshape(-1, self.seq_len*self.seq_embedding)
        labels = self.fc1_seq_enc(labels)

        # Input Transform
        # x = x.permute(0,2,1) # (B,3,S)
        # t_net_transform  = self.input_t_net(x)
        # x = torch.bmm(t_net_transform, x)
        
        # # Shared MLP
        # x = self.bn1(self.relu(self.shared_conv1d_1(x)))
        # x = self.bn2(self.relu(self.shared_conv1d_2(x)))

        # # Feature Transform
        # t_net_feature_transform = self.feature_t_net(x)
        # x = torch.bmm(t_net_feature_transform, x)
        
        # # Shared MLP
        # x = self.bn3(self.relu(self.shared_conv1d_3(x)))
        # x = self.bn4(self.relu(self.shared_conv1d_4(x)))
        
        # x = self.bn5(self.relu(self.shared_conv1d_5(x)))

        x = self.relu(self.bn1(self.conv1(x.permute(0,2,1))))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        global_features = self.max_pool(x).squeeze()
        global_features = torch.cat((global_features, labels), dim = -1)

        x_mu = self.relu(self.fc1_enc_mu(global_features))
        x_logvar = self.relu(self.fc1_enc_logvar(global_features))

        reparam_z = self.reparametrisation(x_mu, x_logvar)

        return reparam_z.squeeze(), x_mu.squeeze(), x_logvar.squeeze()

    def decode(self, z):
        
        z = self.relu(self.fc1_dec(z))
        z = self.relu(self.fc2_dec(z))
        z = self.relu(self.fc3_dec(z))
        z = self.fc4_dec(z)
     
        logit = z.reshape(-1,self.seq_len, self.amino_acids)
        z = self.soft(logit)
        return z, logit

    def reparametrisation(self, x_mu, x_logvar):

        std = torch.exp(0.5 * x_logvar)
        eps = torch.randn_like(std) 
        z_new = x_mu + eps*(std)

        return z_new
    
    def ELBO(self, x, logit, x_mu, x_logvar):


        rec_loss = self.reconstruction_loss_weight*torch.nn.functional.cross_entropy(logit.permute(0,2,1),x, reduction='sum', ignore_index=20)
        KL_loss =self.beta*(-0.5 * torch.sum(1 + x_logvar - x_mu.pow(2) - x_logvar.exp()))
        return (rec_loss + KL_loss) / x.size(0), rec_loss/ x.size(0), KL_loss/ x.size(0)
    
    # def chamfer_distance(self,x,x_reconstructed):

    #     pairwise_dist = torch.cdist(x, x_reconstructed, p = 2)
    #     loss = torch.sum(torch.min(pairwise_dist, dim= -1)[0], dim = -1) + torch.sum(torch.min(pairwise_dist, dim= -2)[0], dim = -1)
        
    #     return loss/2
    
    # def orthogonal_transform_regulariser(self, feature_transform, input_transform):

    #     loss1 = torch.nn.functional.mse_loss(torch.bmm(
    #         input_transform, input_transform.transpose(1, 2)), self.ident1.expand_as(input_transform))
    #     loss2 = torch.nn.functional.mse_loss(torch.bmm(
    #         feature_transform, feature_transform.transpose(1, 2)), self.ident2.expand_as(feature_transform))
        
    #     return (loss1+loss2)/2
    
    def training_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        
        loss, rec_loss, KL_loss = self.ELBO(x[:,:,3], logit, x_mu, x_logvar)

        # transform_loss = self.orthogonal_transform_regulariser(feature_transform, input_transform)
        self.log("train_elbo_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_rec_loss", rec_loss, on_epoch=True, on_step=True, prog_bar=True)
        self.log("train_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
        # self.log("transform_train_loss", transform_loss, on_epoch=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):

        x = batch
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss, rec_loss, KL_loss = self.ELBO(x[:,:,3], logit, x_mu, x_logvar)
        # transform_loss = self.orthogonal_transform_regulariser(feature_transform, input_transform)

        self.log("val_elbo_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_rec_loss", rec_loss, on_epoch=True, on_step=False, prog_bar=True)
        self.log("val_KL_loss", KL_loss, on_epoch=True, on_step=False, prog_bar=True)
        # self.log("transform_val_loss", transform_loss, on_epoch=True)
        # self.log("val_loss", loss+transform_loss, on_epoch=True)
        return loss
    
    def configure_optimizers(self):

        optimizer = self.optimizer(self.parameters(), **self.optimizer_param)

        # 🔹 Using ReduceLROnPlateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        return  {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_elbo_loss_epoch",  # Reduce LR based on validation loss
                "interval": "epoch",  # Step every epoch
                "frequency": 1,  
            }
        }
    
    def on_train_epoch_end(self):
        self.log("learning_rate", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)