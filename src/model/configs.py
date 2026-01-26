from dataclasses import dataclass
from utils.common import load_yaml
import torch

@dataclass
class GPTConfig:
    '''
    Configuration class for GPT model hyperparameters.
    '''
    vocab_size: int 
    block_size: int 
    n_layers: int 
    n_heads: int 
    n_embd: int 
    dropout: float 
    device: str 
    text_bin_path: str 
    model_save_path: str 
    training: bool 
    max_iters: int
    eval_interval: int
    learning_rate: float 
    batch_size: int 
    eval_iters: int 
    
    @classmethod
    def from_yaml(cls, path: str):
        config = load_yaml(path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return cls(
            vocab_size=config['training']['vocab_size'],
            block_size=config['training']['block_size'],
            n_layers=config['training']['n_layers'],
            n_heads=config['training']['n_heads'],
            n_embd=config['training']['n_embd'],
            dropout=config['training']['dropout'],
            device=device,
            text_bin_path=config['path']['text_bin_path'],
            model_save_path=config['path']['model_save_path'],
            training=config['training']['training'],
            max_iters=config['eval']['max_iters'],
            eval_interval=config['eval']['eval_interval'],
            learning_rate=config['training']['learning_rate'],
            batch_size=config['training']['batch_size'],
            eval_iters=config['eval']['eval_iters']
        )
