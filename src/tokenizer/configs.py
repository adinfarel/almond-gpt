from dataclasses import dataclass
from utils.common import load_yaml


@dataclass
class TokenizerConfig:
    '''
    Configuration class for tokenization 
    '''
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