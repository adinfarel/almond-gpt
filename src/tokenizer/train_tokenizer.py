import os
import sys
from tokenizer.bpe import AlmondTokenizer
from dataclasses import dataclass
from utils.common import load_yaml
from utils.logger import logging
from utils.exception import CustomException
# ----------------------------------

@dataclass
class TokenizerConfig:
    vocab_size: int
    vocab_path: str
    merges_path: str
    text_path: str
    
    @classmethod
    def from_yaml(cls, path: str):
        config = load_yaml(path)
        return cls(
            vocab_size=config['vocab_size'],
            vocab_path=config['paths']['vocab_output'],
            merges_path=config['paths']['merges_output'],
            text_path=config['paths']['text_path']
        )


def train_tokenizer():
    '''
    Train the BPE tokenizer and save the vocabulary and merges.
    '''
    # Initialize tokenizer configuration
    tokenizer_config = TokenizerConfig.from_yaml("configs/tokenizer_config.yaml")
    # Load training text
    try:
        with open(tokenizer_config.text_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logging.info(f"Training text loaded from {tokenizer_config.text_path} successfully.")
    except Exception as e:
        raise CustomException(e, sys)
           
    # Initialize tokenizer
    tokenizer = AlmondTokenizer(
        text=text,
        vocab_size=tokenizer_config.vocab_size
    )
    tokenizer.train() # Train the tokenizer
    # Save vocabulary and merges
    tokenizer.save(tokenizer_config.vocab_path, tokenizer_config.merges_path)
    logging.info("Tokenizer training completed and files saved.")
    
# Only run if TRAIN is True 
TRAIN = True
if __name__ == "__main__" and TRAIN:
    train_tokenizer()