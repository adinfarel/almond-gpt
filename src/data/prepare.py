import sys
import os
import torch
from utils.logger import logging
from utils.exception import CustomException
from utils.common import save_bin
from tokenizer.bpe import AlmondTokenizer
from data.dataset import DatasetConfig
# ----------------------------------

def prepare_dataset():
    '''
    Prepare dataset by tokenizing and saving processed data.
    return: binary file path
    '''
    # Initialize tokenizer and dataset config
    dataset_config = DatasetConfig()
    tokenizer = AlmondTokenizer()
    
    # Load vocab and merges
    try:
        tokenizer.load(
            vocab_path=dataset_config.tokenizer_vocab_path,
            merges_path=dataset_config.tokenizer_merges_path
        )
        logging.info("Tokenizer vocabulary and merges loaded successfully.")
    except Exception as e:
        raise CustomException(e, sys)

    # Read raw data
    try:
        with open(dataset_config.raw_data_path, 'r', encoding='utf-8') as file:
            text = file.read()
            logging.info(f"Raw data loaded from {dataset_config.raw_data_path} successfully.")
    except Exception as e:
        raise CustomException(e, sys)
    
    # Tokenize data
    try:
        token_ids = tokenizer.encode(text)
        logging.info("Data tokenization completed successfully.")
    except Exception as e:
        raise CustomException(e, sys)
    
    # Save processed data
    try:
        save_bin(token_ids, dataset_config.processed_data_path)
        logging.info(f"Processed data saved at {dataset_config.processed_data_path} successfully.")
    except Exception as e:
        raise CustomException(e, sys)