import os
import wandb
import torch
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt


def init_wandb(
    project_name: str,
    run_name: str,
    config: dict,
) -> None:
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config
    )    
    
    
def log_figure(
    fig: plt.Figure, 
    name: str
) -> None:
    
    wandb_image = wandb.Image(fig)
    wandb.log({name: wandb_image})
    

def log_table(
    df,
    table_name: str
) -> None:

    wandb.log({table_name: wandb.Table(dataframe=df)})


def log_singular_value(
    value: float,
    name: str
) -> None:
    
    wandb.log({name: value})
    

def log_losses(
    train_loss: float,
    valid_loss: float
) -> None:

    log_dict = {
        'train_loss': train_loss,
        'valid_loss': valid_loss
    }
    
    wandb.log(log_dict)
    

def save_model_architecture(
    model: torch.nn.Module, 
    save_path: str
) -> None:

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model_str = str(model)
    
    with open(save_path+'arch.txt', 'w') as f:
        f.write(model_str)
    wandb.save(save_path+'arch.txt')
    
    wandb.summary['model_architecture'] = model_str
    
    
def finish_run():
    wandb.finish()