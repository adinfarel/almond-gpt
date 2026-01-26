import sys
import os
import torch
import yaml
import math
import torch.nn as nn
import torch.nn.functional as F
import mlflow
import numpy as np
from tqdm import tqdm
from utils.logger import logging
from utils.common import load_model, save_model, load_yaml, load_json, load_bin
from utils.exception import CustomException
from tokenizer.bpe import AlmondTokenizer
from model.gpt import AlmondGPTModel, get_batch, eval_loss
from model.configs import GPTConfig
from tokenizer.configs import TokenizerConfig
from tokenizer.bpe import AlmondTokenizer
from dataclasses import dataclass
from data.prepare import prepare_dataset
# ----------------------------------

@dataclass
class TrainerConfig:
    model_config_path: str = "configs/models_config.yaml"
    tokenizer_config_path: str = "configs/tokenizer_config.yaml"
    data_path: str = "data/processed/text.bin"
    training: bool = True
    experiment_name: str = "AlmondGPT_Experiment_1.0"

def train_model():
    '''
    Train the GPT model.
    '''
    # Load configurations
    trainer_config = TrainerConfig()
    model_config = GPTConfig.from_yaml(trainer_config.model_config_path)
    tokenizer_config = TokenizerConfig.from_yaml(trainer_config.tokenizer_config_path)
    logging.info("Configurations loaded successfully.")
    logging.info("Using device: " + model_config.device)
    
    # Initialize MLflow experiment
    mlflow.set_experiment(trainer_config.experiment_name)
    
    # Get vocab and merges
    if not os.path.exists(tokenizer_config.vocab_path) or not os.path.exists(tokenizer_config.merges_path):
        logging.info("Tokenizer files not found. Want to do training? (y/n).")
        if input().lower() == 'y':
            # Load training (raw) text
            try:
                with open(tokenizer_config.text_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    logging.info(f"Training text loaded from {tokenizer_config.text_path} successfully.")
            except Exception as e:
                raise CustomException(e, sys)
            # Initialize tokenizer
            tokenizer = AlmondTokenizer(text=text, vocab_size=tokenizer_config.vocab_size)
            # Train tokenizer
            tokenizer.train()
            # Save vocabulary and merges
            tokenizer.save(tokenizer_config.vocab_path, tokenizer_config.merges_path)
            logging.info("Tokenizer trained and files saved successfully.")

    # Prepare dataset
    if not os.path.exists(model_config.text_bin_path):
        data_path = prepare_dataset()
    logging.info("Dataset prepared successfully.")
    # Load tokenized data
    tokenized_data = load_bin(model_config.text_bin_path)
    tokenized_data = torch.from_numpy(tokenized_data.astype(np.int64))
    logging.info("Tokenized data loaded successfully.")
    # Initialize model and optimizer
    model = AlmondGPTModel().to(model_config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config.learning_rate, weight_decay=0.1)
    max_lr = model_config.learning_rate
    min_lr = max_lr * 0.1
    warmup_iters = 500
    lr_decay_iters = model_config.max_iters
    
    def get_lr(steps):
        '''
        Get learning rate according to step
        '''
        if step < warmup_iters:
            return max_lr * step / warmup_iters
        
        if step > lr_decay_iters:
            return min_lr
        
        # Cosine decay
        decay_ratio = (step - warmup_iters) / (lr_decay_iters - warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return min_lr + coeff * (max_lr - min_lr)
    
    logging.info("Model and optimizer initialized successfully.")
    # Training loop (to be implemented)
    logging.info("Training started and tracking with MLFlow...")
    with mlflow.start_run():
        # Log parameters
        params = {
            "n_layers": model_config.n_layers,
            "n_heads": model_config.n_heads,
            "n_embd": model_config.n_embd,
            "dropout": model_config.dropout,
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "max_iters": model_config.max_iters,
            "eval_interval": model_config.eval_interval,
            "block_size": model_config.block_size,
            "eval_iters": model_config.eval_iters
        }
        # Track parameters with MLflow
        mlflow.log_params(params)
        model.train()
        
        
        pbar = tqdm(range(model_config.max_iters), desc="Training Progress")
        for step in pbar:
            # Update learning rate per-step
            lr = get_lr(step)
            for param in optimizer.param_groups:
                param['lr'] = lr
            
            mlflow.log_metric("lr", lr, step=step)
            if step % model_config.eval_interval == 0:
                val_loss = eval_loss(model, tokenized_data, model_config)
                print(f"Step {step}: Validation Loss = {val_loss}")
                logging.info(f"Step {step}: Validation Loss = {val_loss}")
                
                # Log metrics with MLflow
                mlflow.log_metric("val_loss", val_loss, step=step)
            
            # Get batch
            xb, yb = get_batch(tokenized_data, model_config.batch_size, model_config.block_size) # get a batch
            xb, yb = xb.to(model_config.device), yb.to(model_config.device) # move to device
            # Forward pass
            logits, loss = model(xb, yb)
            # Zero gradients (don't forget to set them to zero before backward pass)
            optimizer.zero_grad(set_to_none=True)
            # Backward pass
            loss.backward()
            # Clipping gradient
            torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
            # Update parameters
            optimizer.step()
            # Log training loss
            print("Train Loss: ", loss.item())
            mlflow.log_metric("train_loss", loss.item(), step=step)
            pbar.set_postfix({'train_loss': loss.item()})
            
    logging.info("Training completed successfully.")
    # Save the trained model
    save_model(model, model_config.model_save_path)
    print(f"Model saved at {model_config.model_save_path} successfully.")
        
if __name__ == "__main__":
    trainer_config = TrainerConfig()
    if trainer_config.training:
        train_model()