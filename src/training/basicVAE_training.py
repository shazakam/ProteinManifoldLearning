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


def objective(trial, seq_train_dataloader, seq_val_dataloader, max_seq_len, dataset_name):
    # Sweep
    latent_dim_suggestion = trial.suggest_categorical("latent_dim_suggestion", [2]) #16, 32,64, 96, 128, 160, 192])
    hidden_dim_suggestion = trial.suggest_categorical("hidden_dim_suggestion", [512]) #[256,512, 800, 1024])
    beta = trial.suggest_categorical("beta_suggestion", [0.005])#[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1])

    # beta_suggestion = trial.suggest_categorical("beta_suggestion", [0.1, 0.5, 1, 2, 10])

    # BETA ANNEALING EXPERIMENT PARMATERS - Linear Annealing
    # beta = 0.01
    # latent_dim_suggestion = 16
    # hidden_dim_suggestion = 512

    beta_increment_suggestion = 0 #trial.suggest_categorical("beta_increment_suggestion", [0.05, 0.1, 0.5, 1, 2, 5])

    # LATENT DIM EXPERIMENT 
    # latent_dim_suggestion = trial.suggest_categorical("latent_dim_suggestion", [16, 32, 64, 96, 128, 160, 192, 256, 512])
    # hidden_dim_suggestion =  512
    # beta_increment_suggestion = 1

    # HIDDEN DIM EXPERIMENT
    # latent_dim_suggestion = 64
    # hidden_dim_suggestion = trial.suggest_categorical("hidden_dim_suggestion", [64, 128, 256, 512, 800, 1024])
    # beta_increment_suggestion = 1

    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    monitor='epoch',
    save_top_k=1,
    mode = 'max',
    dirpath=f'trained_models/{dataset_name}/optimise_bvae/{trial.study.study_name}/',  # Folder to save checkpoints
    filename=f'{trial.number}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{beta}_BetaInc{beta_increment_suggestion}',   # Checkpoint file name
    )

    # Early Stopping to avoid overfitting
    # early_stop_callback = EarlyStopping(
    # monitor="val_loss_epoch",  # Metric to track
    # mode="min",           # Stop when "val/loss" is minimized
    # patience = 20,           # Wait 15 epochs before stopping
    # verbose=True
    # )   

    # Define Model and Trainer
    log_dir = f'experiments/training_logs/latent_BVAE/{trial.study.study_name}'
    trainer = pl.Trainer(max_epochs = 100,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'BVAE_{trial.number}_LD{latent_dim_suggestion}_HD{hidden_dim_suggestion}_Beta{beta}_BetaInc{beta_increment_suggestion}'),
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
                        beta = beta,
                        beta_cycle = 20,
                        beta_epoch_start = 20,
                        beta_increment = beta_increment_suggestion,
                        dropout = 0,
                        reconstruction_loss_weight = 1)

    
    trainer.fit(model, seq_train_dataloader, seq_val_dataloader)

    
    # Return the final training loss
    return trainer.callback_metrics.get("val_loss_epoch", torch.tensor(float("inf"))).item()


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
    n_trials = 1
    # Create data subsets
    train_subset = SequenceDataset(Subset(dataset, train_idx), max_seq_len)
    val_subset = SequenceDataset(Subset(dataset, val_idx), max_seq_len)
    seq_train_dataloader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    seq_val_dataloader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=True)

    # Get current time
    current_time = datetime.datetime.now()

    # Format the time as hour-day-month
    formatted_time = current_time.strftime("%H-%d-%m")

    # Run Optuna study
    print('Creating Study')
    study = optuna.create_study(study_name=f'{formatted_time}_{experiment_type}_{dataset_name}_BasicVAE_study_BS{BATCH_SIZE}_MS{max_seq_len}_trials{n_trials}', direction="minimize")
    study.optimize(lambda trial: objective(trial, seq_train_dataloader=seq_train_dataloader, 
                                           seq_val_dataloader = seq_val_dataloader, 
                                           max_seq_len = max_seq_len, 
                                           dataset_name = dataset_name), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    importance = optuna.importance.get_param_importances(study)
    df_importance = pd.DataFrame(importance.items(), columns=["Hyperparameter", "Importance"])
    csv_filename = f'experiments/training_logs/latent_BVAE/{study.study_name}/{study.study_name}_himportance.csv'
    df_importance.to_csv(csv_filename, index=False)

    # Convert study results to a DataFrame
    df_results = study.trials_dataframe()

    # Save to CSV
    csv_filename = f"experiments/training_logs/latent_BVAE/{study.study_name}/{study.study_name}_study_results.csv"
    df_results.to_csv(csv_filename, index=False)

    print(f"Saved hyperparameter importance to {csv_filename}")

