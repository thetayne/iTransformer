import itertools
import random
from sklearn.model_selection import ParameterGrid
import os

# Define the hyperparameter space
param_grid = {
    'batch_size': [8, 16, 32],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'train_epochs': [10, 20, 30],
    'embedding_dim': [64, 128, 256, 512],
    'dropout': [0.1, 0.3, 0.5],
    'num_blocks': [2, 3, 7],
    'slstm_at': 1,
    'conv1d_kernel_size': [3, 5, 7],
    'qkv_proj_blocksize': [2, 4, 8],
    'num_heads_xlstm': [2, 4, 8],
}

# Number of random samples to draw
num_samples = 10

# Sample random combinations
random_combinations = list(ParameterGrid(param_grid))
random.shuffle(random_combinations)
sampled_combinations = random_combinations[:num_samples]

# Function to run the model with given hyperparameters
def run_experiment(params, run_number):
    # Construct the command with given hyperparameters
    model_id = f"ECL_96_96_{run_number}_emb_{params['embedding_dim']}"
    command = (
        f"python -u run.py "
        f"--is_training 1 "
        f"--root_path ../dataset/electricity/ "
        f"--data_path electricity.csv "
        f"--model_id {model_id} "
        f"--model xLSTM "
        f"--data custom "
        f"--features M "
        f"--seq_len 96 "
        f"--pred_len 96 "
        f"--embedding_dim {params['embedding_dim']} "
        f"--batch_size {params['batch_size']} "
        f"--learning_rate {params['learning_rate']} "
        f"--train_epochs {params['train_epochs']} "
        f"--enc_in 321 "
        f"--c_out 321 "
        f"--kernal_size {params['conv1d_kernel_size']} "
        f"--num_heads_xlstm {params['num_heads_xlstm']} "
        f"--qkv_proj_blocksize {params['qkv_proj_blocksize']} "
        f"--dropout {params['dropout']} "
        f"--num_blocks {params['num_blocks']} "
        f"--weight_decay {params['weight_decay']} "
        f"--slstm_at {params['slstm_at']} "
    )
    # Run the command
    os.system(command)

# Run experiments with sampled combinations
for i, params in enumerate(sampled_combinations, start=1):
    run_experiment(params, i)
