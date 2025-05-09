import yaml
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from src.models.PointNetVae_chamfer_split import PointNetVAE
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import TensorDataset
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset, GeneOntologyDataset, ProteinFamilyDataset
import sys
import random
import numpy as np
import pandas as pd

def BetaExperiment(point_train_dataloader, point_val_dataloader, dataset_name):
    beta_inc = 0 #[0.0001, 0.001, 0.01, 0.1]
    latent_dim = 64
    global_feature_size = 512
    conv_hidden = 8
    hidden_dim = 512
    starting_beta = [0.001, 0.01, 0.1, 1]

    for idx, start_beta in enumerate(starting_beta):
        # Model Checkpoints and saving
        checkpoint_callback = ModelCheckpoint(
        monitor='val_loss_epoch',
        save_top_k=1,
        mode = 'min',
        dirpath=f'trained_models/{dataset_name}/BETA_point_vae/{dataset_name}_BETA_EXP/',  # Folder to save checkpoints
        filename=f'{idx}_LD{latent_dim}_GF{global_feature_size}_Beta{start_beta}',   # Checkpoint file name
        )

        # Define Model and Trainer
        log_dir = f'experiments/training_logs/latent_PointVAE/BETA_EXP'
        trainer = pl.Trainer(max_epochs = 100,
            accelerator="auto",
            devices="auto",
            logger=TensorBoardLogger(save_dir=log_dir, name= f'PVAE_{idx}_LD{latent_dim}_GF{global_feature_size}_Beta{start_beta}_CH{conv_hidden}'),
            callbacks=[checkpoint_callback],
            log_every_n_steps = 20
            )
        
        # Initialise Optimizer, Model anad begin training
        optimizer = torch.optim.AdamW
        optimzer_param = {'lr':0.001}

        model = PointNetVAE(latent_dim = latent_dim,
                            optimizer = optimizer,
                            optimizer_param = optimzer_param,
                            global_feature_size = global_feature_size, 
                            beta=start_beta,
                            beta_increment=beta_inc,
                            conv_hidden_dim = conv_hidden,
                            hidden_dim = hidden_dim)

        
        trainer.fit(model, point_train_dataloader, point_val_dataloader)

def final_model_training(point_train_dataloader, point_val_dataloader, dataset_name):
    beta_inc = 0
    latent_dim = 64
    global_feature_size = 512
    conv_hidden = 8
    hidden_dim = 512
    starting_beta = 0.001

    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_epoch',
    save_top_k=1,
    mode = 'min',
    dirpath=f'trained_models/{dataset_name}/PVAE/FINAL_MODEL/',  # Folder to save checkpoints
    filename=f'FINAL_PVAE_LD{latent_dim}_GF{global_feature_size}_BetaInc{beta_inc}_Beta{starting_beta}_HD{hidden_dim}_CH{conv_hidden}',   # Checkpoint file name
    )

    # Define Model and Trainer
    log_dir = f'experiments/training_logs/latent_PointVAE/FINAL_MODEL'
    trainer = pl.Trainer(max_epochs = 200,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'PVAE_FINAL_MODEL_LD{latent_dim}_GF{global_feature_size}_BetaInc{beta_inc}_Beta{starting_beta}_HD{hidden_dim}_CH{conv_hidden}'),
        callbacks=[checkpoint_callback],
        log_every_n_steps = 20
        )
    
    # Initialise Optimizer, Model anad begin training
    optimizer = torch.optim.AdamW
    optimzer_param = {'lr':0.0001}

    model = PointNetVAE(latent_dim = latent_dim,
                        optimizer = optimizer,
                        optimizer_param = optimzer_param,
                        global_feature_size = global_feature_size, 
                        beta=starting_beta,
                        beta_increment=beta_inc,
                        conv_hidden_dim = conv_hidden,
                        hidden_dim = hidden_dim)

    
    trainer.fit(model, point_train_dataloader, point_val_dataloader)

# Main function to run experiments
if __name__ == "__main__":
    # config = load_config("src/training/training_config.yaml")

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

    # Create data subsets
    train_subset = TensorDataset(torch.load('data/processed/point/Pfam_Point_norm_proc/Pfam_data_train_norm.pt'))
    val_subset = TensorDataset(torch.load('data/processed/point/Pfam_Point_norm_proc/Pfam_data_val_norm.pt'))

    point_train_dataloader = DataLoader(train_subset, batch_size = 128)
    point_val_dataloader = DataLoader(val_subset, batch_size = 128)

    print('---- LENGTH POINT TRAIN DATALOADER ----')
    print(f'-----{len(point_train_dataloader)}------')

    print('---- LENGTH POINT VAL DATALOADER ----')
    print(f'-----{len(point_val_dataloader)}------')

    if len(point_val_dataloader) >= len(point_train_dataloader):
        print('POTENTIAL DATALEAK CHECK DATALOADERS')
        sys.exit()

    

    if experiment_type == 'Beta':
        BetaExperiment(point_train_dataloader, point_val_dataloader, dataset_name)

    elif experiment_type == 'Final_Model':
        np.save('point_val_indices.npy', val_idx)
        np.save('point_train_indices.npy', train_idx)
        final_model_training(point_train_dataloader, point_val_dataloader, dataset_name)
    else:
        exit
    