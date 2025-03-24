import yaml
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from ..models.basicVae import LitBasicVae 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..dataset_classes.sequenceDataset import *
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset, GeneOntologyDataset, ProteinFamilyDataset
import sys
import random
import optuna
import pandas as pd
import datetime


def BetaExperiment(seq_train_dataloader, seq_val_dataloader, dataset_name):
    latent_dim_suggestion = 16
    hidden_dim_suggestion = 512
    beta_increments = [0.05, 0.1, 0.5, 1, 2, 5]

    for idx, beta_inc in enumerate(beta_increments):
        # Model Checkpoints and saving
        checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        save_top_k=1,
        mode = 'max',
        dirpath=f'trained_models/{dataset_name}/BVAE/BETA_EXP/',  # Folder to save checkpoints
        filename=f'{idx}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{beta_inc}',   # Checkpoint file name
        ) 

        # Define Model and Trainer
        log_dir = f'experiments/training_logs/latent_BVAE/BETA_EXP'
        trainer = pl.Trainer(max_epochs = 100,
            accelerator="auto",
            devices="auto",
            logger=TensorBoardLogger(save_dir=log_dir, name= f'BVAE_{idx}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{beta_inc}'),
            callbacks=[checkpoint_callback],
            log_every_n_steps = 20
            )
        
        # Initialise Optimizer, Model anad begin training
        optimizer = torch.optim.AdamW
        optimzer_param = {'lr':0.001}

        model = LitBasicVae(latent_dim = latent_dim_suggestion, 
                            optimizer = optimizer, 
                            optimizer_param = optimzer_param,
                            seq_len = max_seq_len, 
                            amino_acids = 21, 
                            hidden_dim = hidden_dim_suggestion,
                            beta = 0.01,
                            beta_cycle = 20,
                            beta_epoch_start = 20,
                            beta_increment = beta_inc,
                            dropout = 0,
                            reconstruction_loss_weight = 1)

        
        trainer.fit(model, seq_train_dataloader, seq_val_dataloader)


# Main function to run experiments
if __name__ == "__main__":

    dataset_name = input('Dataset ')
    experiment_type = input('Experiment type ')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load Data
    if dataset_name== 'ProteinLigand':
        dataset = ProteinLigandInterfaceDataset(root='data').to_point().torch()
    elif dataset_name == 'AlphaFold':
        dataset = AlphaFoldDataset(root='data').to_point().torch()
    elif dataset_name == 'GO':
        dataset = GeneOntologyDataset(root='data').to_point().torch()
    elif dataset_name == 'Pfam':
        dataset = ProteinFamilyDataset(root='data').to_point().torch()
    else:
        print('Other datasets not used at the moment')
        sys.exit()

    max_seq_len = 500
    idx_list = range(len(dataset))
    subset_size = int(len(dataset)//10)
    val_idx = random.sample(idx_list, subset_size)  # Get random subset
    train_idx = list(set(idx_list) - set(val_idx))

    BATCH_SIZE = 128
    n_trials = 30
    # Create data subsets
    train_subset = SequenceDataset(Subset(dataset, train_idx), max_seq_len)
    val_subset = SequenceDataset(Subset(dataset, val_idx), max_seq_len)
    seq_train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    seq_val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)

    # Get current time
    current_time = datetime.datetime.now()

    # Format the time as hour-day-month
    formatted_time = current_time.strftime("%H-%d-%m")

    if experiment_type == 'Beta':
        BetaExperiment(seq_train_dataloader, seq_val_dataloader, dataset_name)
    else:
        exit

    
