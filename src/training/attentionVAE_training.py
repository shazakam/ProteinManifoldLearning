import yaml
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from ..models.attentionVae import AttentionVAE 
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from ..dataset_classes.sequenceDataset import *
import pytorch_lightning as pl
from proteinshake.datasets import ProteinLigandInterfaceDataset, AlphaFoldDataset, GeneOntologyDataset
import sys
import random
import optuna

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

def objective(trial, seq_train_dataloader, seq_val_dataloader, max_seq_len):
    latent_dim_suggestion = trial.suggest_categorical("latent_dim_suggestion", [192]) #trial.suggest_int("latent_dim_suggestion", 64, 256, step=64)
    hidden_dim_suggestion = trial.suggest_categorical("hidden_dim_suggestion", [512]) #trial.suggest_int("hidden_dim_suggestion", 256, 512, step=128)
    dropout_suggestion = trial.suggest_categorical("dropout_suggestion", [0,0.6]) #trial.suggest_float("dropout_suggesstion",0,0.6, step = 0.1)
    num_heads_suggestion = trial.suggest_categorical("num_heads_suggestion", [2,8])
    embed_dim_suggestion = trial.suggest_categorical("embed_dim_suggestion", [128]) #trial.suggest_int("embed_dim_suggestion",64,256, step = 64)

    # Model Checkpoints and saving
    checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    save_top_k=1,
    mode = 'min',
    dirpath=f'trained_models/GO/optimise_attention_vae/{trial.study.study_name}/',  # Folder to save checkpoints
    filename=f'{trial.study.study_name}_{trial.number}',   # Checkpoint file name
    )

    # Early Stopping to avoid overfitting
    early_stop_callback = EarlyStopping(
    monitor="val_loss_epoch",  # Metric to track
    mode="min",           # Stop when "val/loss" is minimized
    patience = 5,           # Wait 5 epochs before stopping
    verbose=True
    )   

    # Define Model and Trainer
    log_dir = f'experiments/training_logs/latent_attentionVAE/{trial.study.study_name}'
    trainer = pl.Trainer(max_epochs = 100,
        accelerator="auto",
        devices="auto",
        logger=TensorBoardLogger(save_dir=log_dir, name= f'optimise_attention_vae_trial_{trial.number}'),
        callbacks=[early_stop_callback, checkpoint_callback],
        log_every_n_steps = 20
        )
    
    # Initialise Optimizer, Model anad begin training
    optimizer = torch.optim.AdamW
    optimzer_param = {'lr':0.001}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AttentionVAE(optimizer, optimzer_param, 
                         embed_dim=embed_dim_suggestion, 
                         hidden_dim=hidden_dim_suggestion,
                         num_heads=num_heads_suggestion, 
                         dropout=dropout_suggestion,
                         latent_dim=latent_dim_suggestion, 
                         seq_len=max_seq_len)

    
    trainer.fit(model, seq_train_dataloader, seq_val_dataloader)

    
    # Return the final training loss
    return trainer.callback_metrics.get("val_loss_epoch", torch.tensor(float("inf"))).item()

# Main function to run experiments
if __name__ == "__main__":
    config = load_config("src/training/training_config.yaml")

    # Specify Experiment / Model Training Configs
    attention_vae_exp = input('Enter: <experiment_type>, <experiment_name> ')
    attention_vae_exp = attention_vae_exp.split(',')
    exp_config = config['AttentionVAE'][attention_vae_exp[0]][attention_vae_exp[1]]
    model_name = exp_config['model']
    # Set random seed for reproducibility
    torch.manual_seed(exp_config["seed"])
    
    # Load Data
    if exp_config['dataset'] == 'ProteinLigand':
        if exp_config['data_type'] == 'point':
            dataset = ProteinLigandInterfaceDataset(root='data').to_point().torch()
    elif exp_config['dataset'] == 'AlphaFold':
        if exp_config['data_type'] == 'point':
            dataset = AlphaFoldDataset(root='data').to_point().torch()
    elif exp_config['dataset'] == 'GO':
        if exp_config['data_type'] =='point':
            dataset = GeneOntologyDataset(root='data').to_point().torch()
    else:
        print('Other datasets not used at the moment')
        sys.exit()

    max_seq_len = exp_config['max_seq_len']
    idx_list = range(len(dataset))
    subset_size = int(len(dataset)//10)
    val_idx = random.sample(idx_list, subset_size)  # Get random subset
    train_idx = list(set(idx_list) - set(val_idx))

    # Create data subsets
    train_subset = SequenceDataset(Subset(dataset, train_idx), max_seq_len, transformer_input=True)
    val_subset = SequenceDataset(Subset(dataset, val_idx), max_seq_len, transformer_input=True)
    seq_train_dataloader = DataLoader(train_subset, batch_size=exp_config['batch_size'], shuffle=True)
    seq_val_dataloader = DataLoader(val_subset, batch_size=exp_config['batch_size'], shuffle=True)
    dataset = exp_config['dataset']

    print(exp_config)
    
    # # Model Checkpoints and saving
    # checkpoint_callback = ModelCheckpoint(
    #     monitor='val_loss',
    #     save_top_k=1,
    #     mode = 'min',
    #     dirpath=f'trained_models/{dataset}/trained_attention_vae/',  # Folder to save checkpoints
    #     filename=f'attention_vae_{attention_vae_exp[0]}_{attention_vae_exp[1]}_{dataset}',   # Checkpoint file name
    # )

#     # Early Stopping to avoid overfitting
#     early_stop_callback = EarlyStopping(
#     monitor="val_loss",  # Metric to track
#     mode="min",           # Stop when "val/loss" is minimized
#     patience=8,           # Wait 5 epochs before stopping
#     verbose=True
# )

#     # Define Model and Trainer
#     log_dir = 'experiments/training_logs/latent_AttentionVAE'
#     trainer = pl.Trainer(max_epochs=exp_config['epochs'],
#         accelerator="auto",
#         devices="auto",
#         logger=TensorBoardLogger(save_dir=log_dir, name= f'{model_name}'),
#         callbacks=[early_stop_callback, checkpoint_callback],
#         log_every_n_steps = 20
#         )
    
#     # Initialise Optimizer, Model anad begin training
#     optimizer = get_optimizer(exp_config['optimizer'])
#     optimzer_param = exp_config['optimizer_param']
#     model = AttentionVAE(optimizer, optimzer_param, 
#                          exp_config['embed_dim'], exp_config['hidden_dim'],
#                          exp_config['num_heads'], exp_config['dropout'],
#                          exp_config['latent_dim'], max_seq_len
#                          )
#     trainer.fit(model, seq_train_dataloader, seq_val_dataloader)

    print('Creating Study')
    study = optuna.create_study(study_name='AttentionVAE_HyperParam_Tuning_v2_V7', direction="minimize")
    study.optimize(lambda trial: objective(trial, seq_train_dataloader=seq_train_dataloader, seq_val_dataloader=seq_val_dataloader, max_seq_len=max_seq_len), n_trials=20)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))