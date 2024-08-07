import itertools
import random
from sklearn.model_selection import ParameterGrid
import os

# Define the hyperparameter space
param_grid = {
    'batch_size': [8, 16, 32],
    'learning_rate': [1e-4, 5e-4, 1e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'train_epochs': [10, 30, 50],
    'dropout': [0.1, 0.3, 0.5],
    'num_blocks': [2, 3, 7],
    'slstm_at': [1],
    'conv1d_kernel_size': [3, 5, 7],
    'qkv_proj_blocksize': [2, 4, 8],
    'num_heads_xlstm': [2, 4, 8],
    'grad_clip_norm': [0.5, 1.0, 5.0]
}

# Number of random samples to draw for each embedding_dim
num_samples = 10

# Function to run the model with given hyperparameters
def run_experiment(params, run_number):
    # Construct the command with given hyperparameters
    model_id = f"ECL_96_96_{run_number}_emb_{params['embedding_dim']}"
    command = (
        f"python -u iTransformer/run.py "
        f"--is_training 1 "
#       f"--root_path ../data "
        f"--data_path /input-data/electricity.csv "
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
        f"--patience 10 "
        f"--kernal_size {params['conv1d_kernel_size']} "
        f"--num_heads_xlstm {params['num_heads_xlstm']} "
        f"--qkv_proj_blocksize {params['qkv_proj_blocksize']} "
        f"--dropout {params['dropout']} "
        f"--num_blocks {params['num_blocks']} "
        f"--weight_decay {params['weight_decay']} "
        f"--slstm_at {params['slstm_at']} "
        f"--grad_clip_norm {params['grad_clip_norm']} "
    )
    # Run the command
    os.system(command)

# Loop through each embedding_dim and perform random search
embedding_dims = [64, 128, 256, 512]
run_number = 1
for embedding_dim in embedding_dims:
    # Filter the parameter grid to exclude embedding_dim
    param_grid_filtered = {k: v for k, v in param_grid.items() if k != 'embedding_dim'}
    random_combinations = list(ParameterGrid(param_grid_filtered))
    random.shuffle(random_combinations)
    sampled_combinations = random_combinations[:num_samples]
    
    for params in sampled_combinations:
        # Add the current embedding_dim to the params
        params['embedding_dim'] = embedding_dim
        # Run the experiment
        run_experiment(params, run_number)
        run_number += 1
