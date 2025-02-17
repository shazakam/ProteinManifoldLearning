import torch
import torch.nn as nn
import pytorch_lightning as pl

class LitBasicVae(pl.LightningModule):
    def __init__(self, latent_dim, optimizer, optimizer_param, seq_len = 500, amino_acids = 21, hidden_dim=512):
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
        self.latent_dim = latent_dim
        self.optimizer = optimizer
        self.optimizer_param = optimizer_param
        self.seq_len = seq_len
        self.amino_acids = amino_acids
        self.input_dim = self.amino_acids*self.seq_len

        # Activation Functions
        self.relu = nn.ReLU()
        self.soft = nn.Softmax()

        # Encoder 
        self.fc1_enc = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc3_enc_mean = nn.Linear(self.hidden_dim, latent_dim)
        self.fc3_enc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
        self.fc1_dec = nn.Linear(latent_dim,self.hidden_dim)
        self.fc3_dec = nn.Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        """
        Forward pass through the VAE.

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
        logit = self.fc3_dec(z)
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
        eps = torch.randn_like(std) #.to(self.device)
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
        x = x.reshape(-1,self.seq_len,self.amino_acids)
        logit = logit.reshape(-1,self.seq_len,self.amino_acids)
        x_true_indices = x.argmax(dim=-1)
        rec_loss =  torch.nn.functional.cross_entropy(logit.permute(0,2,1),x_true_indices, reduction='sum')

        # rec_loss =  torch.nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        # x_true_indices = x.argmax(dim=-1)
        # print(x_hat.shape)
        # print(x_true_indices.shape)
        # rec_loss = torch.nn.functional.cross_entropy(x_hat, x_true_indices, reduction='sum')
        # rec_loss = torch.nn.functional.mse_loss(x_hat,x,reduction = 'sum')
        
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
        x = batch.view(-1,self.input_dim)

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

        x = batch.view(-1,self.input_dim)
        rep_z, x_mu, x_logvar, x_rec, logit = self(x)
        loss = self.ELBO(x, logit,x_mu, x_logvar)
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
    
    def on_train_epoch_end(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                self.logger.experiment.add_histogram(f"weights/{name}", param, self.current_epoch)

