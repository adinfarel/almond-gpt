from dataclasses import dataclass


class DatasetConfig:
    '''
    Configuration class for dataset paths.
    '''
    raw_data_path: str = "data/raw/input.txt"
    processed_data_path: str = "data/processed/processed_data.bin"
    tokenizer_vocab_path: str = "models/tokenizer/vocab.json"
    tokenizer_merges_path: str = "models/tokenizer/merges.json"