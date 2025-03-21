import yaml
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from ..models.pointNetVae import PointNetVAE 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..dataset_classes.sequenceDataset import *
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset, GeneOntologyDataset, ProteinFamilyDataset
import sys
import random
import optuna
import pandas as pd

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


def objective(trial, seq_train_dataloader, seq_val_dataloader, dataset_name):
    latent_dim_suggestion = trial.suggest_categorical("latent_dim_suggestion", [2, 16, 32, 64, 128, 256])
    embed_dim_suggestion = trial.suggest_categorical("embed_dim_suggestion", [20, 32, 64, 128, 256, 512])
    global_feature_size_suggestion = trial.suggest_categorical("global_feat_suggestion", [128, 256, 512, 1024])
    beta_suggestion = trial.suggest_categorical("beta_suggestion", [1, 5, 10, 20])
    conv_hidden_dim_suggestion = trial.suggest_categorical("conv_hidden_suggestion", [32, 64, 128])
    hidden_dim_suggestion = trial.suggest_categorical("hidden_dim_suggestion", [256, 512, 1024])
    
    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss_epoch',
    save_top_k=1,
    mode = 'min',
    dirpath=f'trained_models/{dataset_name}/optimise_point_vae/{trial.study.study_name}/',  # Folder to save checkpoints
    filename=f'{trial.number}_LD{latent_dim_suggestion}_GF{global_feature_size_suggestion}_Beta{beta_suggestion}',   # Checkpoint file name
    )

    # Early Stopping to avoid overfitting
    early_stop_callback = EarlyStopping(
    monitor="val_loss_epoch",  # Metric to track
    mode="min",           # Stop when "val/loss" is minimized
    patience = 15,           # Wait 5 epochs before stopping
    verbose=True
    )   

    # Define Model and Trainer
    log_dir = f'experiments/training_logs/latent_PointVAE/{trial.study.study_name}'
    trainer = pl.Trainer(max_epochs = 100,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'PVAE_{trial.number}_LD{latent_dim_suggestion}_GF{global_feature_size_suggestion}_Beta{beta_suggestion}_CH{conv_hidden_dim_suggestion}_EM{embed_dim_suggestion}'),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps = 20
        )
    
    # Initialise Optimizer, Model anad begin training
    optimizer = torch.optim.AdamW
    optimzer_param = {'lr':0.001}

    model = PointNetVAE(latent_dim = latent_dim_suggestion,
                        optimizer = optimizer,
                        optimizer_param = optimzer_param,
                        seq_embedding = embed_dim_suggestion, 
                        global_feature_size = global_feature_size_suggestion, 
                        beta=beta_suggestion,
                        conv_hidden_dim = conv_hidden_dim_suggestion,
                        hidden_dim = hidden_dim_suggestion)

    
    trainer.fit(model, seq_train_dataloader, seq_val_dataloader)

    
    # Return the final training loss
    return trainer.callback_metrics.get("val_loss_epoch", torch.tensor(float("inf"))).item()


# Main function to run experiments
if __name__ == "__main__":
    # config = load_config("src/training/training_config.yaml")

    dataset_name = input('Dataset ')

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
    n_trials = 3

    # Create data subsets
    train_subset = SequenceDataset(Subset(dataset, train_idx), max_seq_len)
    val_subset = SequenceDataset(Subset(dataset, val_idx), max_seq_len)
    seq_train_dataloader = DataLoader(train_subset, batch_size=128, shuffle=True)
    seq_val_dataloader = DataLoader(val_subset, batch_size=128, shuffle=True)

    # Run Optuna study
    print('Creating Study')
    study = optuna.create_study(study_name=f'{dataset_name}_PointVAE_study_BS{BATCH_SIZE}_MS{max_seq_len}_trials{n_trials}', direction="minimize")
    study.optimize(lambda trial: objective(trial, seq_train_dataloader=seq_train_dataloader, 
                                           seq_val_dataloader = seq_val_dataloader, 
                                           dataset_name = dataset_name), n_trials=n_trials)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    importance = optuna.importance.get_param_importances(study)
    df_importance = pd.DataFrame(importance.items(), columns=["Hyperparameter", "Importance"])
    csv_filename = f'experiments/training_logs/latent_PointVAE/{study.study_name}/{study.study_name}_himportance.csv'
    df_importance.to_csv(csv_filename, index=False)

    # Convert study results to a DataFrame
    df_results = study.trials_dataframe()

    # Save to CSV
    csv_filename = f"experiments/training_logs/latent_PointVAE/{study.study_name}/{study.study_name}_study_results.csv"
    df_results.to_csv(csv_filename, index=False)

    print(f"Saved hyperparameter importance to {csv_filename}")