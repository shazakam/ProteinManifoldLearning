import yaml
import torch
from torch.utils.data import DataLoader
from ..models.basicVae import LitBasicVae 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ..dataset_classes.sequenceDataset import *
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset
import sys

# Load config
def load_config(config_file="config.yaml"):
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)
    return config

# Function to select optimizer dynamically based on config
def get_optimizer(optimizer):
    if optimizer == 'adam':
        optimizer = torch.optim.Adam
    elif optimizer == 'sgd':
        optimizer = torch.optim.SGD
    elif optimizer == 'adamw':
        optimizer = torch.optim.AdamW
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer}")
    
    return optimizer

# Main function to run experiments
if __name__ == "__main__":
    config = load_config("src/training/training_config.yaml")

    # Specify Experiment / Model Training Configs
    bvae_exp = input('Enter: <experiment_type>, <experiment_name> ')
    bvae_exp = bvae_exp.split(',')
    exp_config = config['basic_vae'][bvae_exp[0]][bvae_exp[1]]
    model_name = exp_config['model']
    # Set random seed for reproducibility
    torch.manual_seed(exp_config["seed"])
    
    # Load Data
    if exp_config['dataset'] == 'ProteinLigand':
        if exp_config['data_type'] == 'point':
            dataset = ProteinLigandInterfaceDataset(root='data').to_point().torch()
    else:
        print('Other datasets not used at the moment, TODO: change config file and implement ')
        sys.exit()

    seq_test_data = SequenceDataset(dataset)
    seq_train_dataloader = DataLoader(seq_test_data, batch_size=exp_config['batch_size'], shuffle=False)
    x_dim = seq_test_data[0].shape[0] # Input Dimensionality of data points

    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    dirpath='trained_models/trained_bvae',  # Folder to save checkpoints
    filename=f'bvae_{bvae_exp[0]}_{bvae_exp[1]}',   # Checkpoint file name
    )

    # Define Model and Trainer
    log_dir = 'experiments/training_logs/latent_BVAE'
    trainer = pl.Trainer(max_epochs=exp_config['epochs'],
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'{model_name}'),
        callbacks=[checkpoint_callback],
        log_every_n_steps = 20
        )
    
    # Initialise Optimizer, Optimizer, Model anad begin training
    optimizer = get_optimizer(exp_config['optimizer'])
    optimzer_param = exp_config['optimizer_param']
    model = LitBasicVae(exp_config['latent_dim'], x_dim,optimizer, optimzer_param, exp_config['hidden_dims'])
    trainer.fit(model, seq_train_dataloader)



