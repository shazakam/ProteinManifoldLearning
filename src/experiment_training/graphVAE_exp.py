import yaml
import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from ..models.graphVAE import GraphVAE 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset, GeneOntologyDataset, ProteinFamilyDataset
import sys
import random
import optuna
from src.dataset_classes.graphDataset import *
import pandas as pd
import datetime
import numpy as np

def BetaExperiment(graph_train_dataloader, graph_val_dataloader, dataset_name):
    latent_dim_suggestion = 32 #
    hidden_dim_suggestion = 512 #
    conv_hidden_dim_suggestion = 16 #
    lr_suggestion = 0.0001 # CHANGE TO FINAL VALUES
    beta_inc = 0 
    starting_beta = [0.001, 0.01, 0.1, 1]
    for idx, start_beta in enumerate(starting_beta):
        # Model Checkpoints and saving
        checkpoint_callback = ModelCheckpoint(
        monitor='epoch',
        save_top_k=1,
        mode = 'max',
        dirpath=f'trained_models/{dataset_name}/GVAE/BETA_EXP/',  # Folder to save checkpoints
        filename=f'{idx}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{start_beta}_GCH{conv_hidden_dim_suggestion}_LR{lr_suggestion}',   # Checkpoint file name
        ) 

        # Define Model and Trainer
        log_dir = f'experiments/training_logs/latent_GraphVAE/BETA_EXP'
        trainer = pl.Trainer(max_epochs = 100,
            accelerator="auto",
            devices="auto",
            logger=TensorBoardLogger(save_dir=log_dir, name= f'GraphVAE_{idx}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{start_beta}_GCH{conv_hidden_dim_suggestion}_LR{lr_suggestion}'),
            callbacks=[checkpoint_callback],
            log_every_n_steps = 20
            )
        

        # Initialise Optimizer, Model anad begin training
        optimizer = torch.optim.AdamW
        optimizer_param = {'lr':lr_suggestion}

        model = GraphVAE(latent_dim = latent_dim_suggestion, 
                        optimizer = optimizer, 
                        optimizer_param = optimizer_param, 
                        seq_len = 500, 
                        amino_acids = 21,
                        conv_hidden_dim = conv_hidden_dim_suggestion, 
                        hidden_dim=hidden_dim_suggestion, 
                        beta = start_beta,
                        beta_increment=beta_inc)

        
        trainer.fit(model, graph_train_dataloader, graph_val_dataloader)


def final_model_training(graph_train_dataloader, graph_val_dataloader, dataset_name):
    latent_dim_suggestion = 32 #
    hidden_dim_suggestion = 512 #
    conv_hidden_dim_suggestion = 16 #
    lr_suggestion = 0.0001 
    starting_beta = 0.005
    beta_inc = 0

    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    monitor='epoch',
    save_top_k=1,
    mode = 'max',
    dirpath=f'trained_models/{dataset_name}/GVAE/FINAL_MODEL/',  # Folder to save checkpoints
    filename=f'LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{starting_beta}_BetaInc{beta_inc}_GCH{conv_hidden_dim_suggestion}_LR{lr_suggestion}',   # Checkpoint file name
    ) 

    # Define Model and Trainer
    log_dir = f'experiments/training_logs/latent_GraphVAE/FINAL_MODEL'
    trainer = pl.Trainer(max_epochs = 200,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'GraphVAE_FINAL_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{starting_beta}_BetaInc{beta_inc}_GCH{conv_hidden_dim_suggestion}_LR{lr_suggestion}'),
        callbacks=[checkpoint_callback],
        log_every_n_steps = 20
        )
    

    # Initialise Optimizer, Model anad begin training
    optimizer = torch.optim.AdamW
    optimizer_param = {'lr':lr_suggestion}

    model = GraphVAE(latent_dim = latent_dim_suggestion, 
                    optimizer = optimizer, 
                    optimizer_param = optimizer_param, 
                    seq_len = 500, 
                    amino_acids = 21,
                    conv_hidden_dim = conv_hidden_dim_suggestion, 
                    hidden_dim=hidden_dim_suggestion, 
                    beta = starting_beta,
                    beta_increment=beta_inc,
                    beta_epoch_start = 20)

    
    trainer.fit(model, graph_train_dataloader, graph_val_dataloader)



# Main function to run experiments
if __name__ == "__main__":

    dataset_name = input('Dataset ')
    experiment_type = input('Experiment type ')

    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Load Data
    if dataset_name== 'ProteinLigand':
        dataset = ProteinLigandInterfaceDataset(root='data').to_graph(eps = 8).pyg()
    elif dataset_name == 'AlphaFold':
        dataset = AlphaFoldDataset(root='data').to_graph(eps = 8).pyg()
    elif dataset_name == 'GO':
        dataset = GeneOntologyDataset(root='data').to_graph(eps = 8).pyg()
    elif dataset_name == 'Pfam':
        dataset = ProteinFamilyDataset(root='data').to_graph(eps = 8).pyg()
    else:
        print('Other datasets not used at the moment')
        sys.exit()

    dataset = load_graph_data(dataset)
    max_seq_len = 500
    idx_list = range(len(dataset))
    subset_size = int(len(dataset)//10)
    val_idx = random.sample(idx_list, subset_size)  # Get random subset
    train_idx = list(set(idx_list) - set(val_idx))

    BATCH_SIZE = 128

    # Create data subsets
    train_datalist = [dataset[idx] for idx in train_idx]
    val_datalist = [dataset[idx] for idx in val_idx]
    train_dataset = GraphListDataset(train_datalist)
    val_dataset  = GraphListDataset(val_datalist)

    graph_train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    graph_val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE, shuffle=False)

    print('---- LENGTH GRAPH TRAIN DATALOADER ----')
    print(f'-----{len(graph_train_dataloader)}------')

    print('---- LENGTH GRAPH VAL DATALOADER ----')
    print(f'-----{len(graph_val_dataloader)}------')

    if len(graph_val_dataloader) >= len(graph_train_dataloader):
        print('POTENTIAL DATALEAK CHECK DATALOADERS')
        sys.exit()

    # Get current time
    current_time = datetime.datetime.now()

    # Format the time as hour-day-month
    formatted_time = current_time.strftime("%H-%d-%m")

    if experiment_type == 'Beta':
        BetaExperiment(graph_train_dataloader, graph_val_dataloader, dataset_name)

    elif experiment_type == 'Final_Model':
        np.save('graph_val_indices.npy', val_idx)
        np.save('graph_train_indices.npy', train_idx)
        final_model_training(graph_train_dataloader, graph_val_dataloader, dataset_name)
    else:
        exit