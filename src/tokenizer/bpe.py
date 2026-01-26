import sys
import os
from typing import List, Tuple
import torch.nn as nn
from utils.logger import logging
from utils.common import save_json, load_json
from utils.exception import CustomException
from tqdm import tqdm
# ----------------------------------

class AlmondTokenizer(nn.Module):
    '''
    A Byte Pair Encoding (BPE) tokenizer implementation.'''
    def __init__(self, text='', vocab_size: int = 768):
        '''
        Initialize the BPE tokenizer.
        :param text: str: The input text to train the tokenizer on.
        :param vocab_size: int: The desired vocabulary size.
        '''
        super().__init__()
        self.vocab = {}
        self.merges = {}
        self.tokens = text.encode('utf-8')
        self.vocab_size = vocab_size
        self.single_byte_size = 256 # default single byte tokens 0 - 255
        # Additional initialization code here

    def train(self):
        '''
        Train the BPE tokenizer.
        :return: None
        '''
        num_merges = self.vocab_size - self.single_byte_size # First 256 are single byte tokens
        ids = self.tokens
        logging.info("Starting BPE training...")
        for i in tqdm(range(num_merges), desc="Training BPE Tokenizer"):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = i + self.single_byte_size
            print(f"Merging pair: {pair} as new token ID: {idx}")
            self.merges[pair] = idx
            ids = self.merge_pair(pair, ids, 256 + i)
        logging.info("Training completed successfully.")
        
    
    def encode(self, text: str) -> List[int]:
        '''
        Encode the input text into a list of token IDs.
        :param text: str: The input text to encode.
        :return: List[int]: The list of token IDs.
        '''
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            best_pair = min(stats, key=lambda pair: self.merges.get(pair, float('inf')))
            if best_pair not in self.merges:
                break
            ids = self.merge_pair(best_pair, ids, self.merges[best_pair])
        logging.info(f"Encoded text to IDs: {ids} successfully.")
        return ids

    def decode(self, token_ids: List[int]) -> str:
        '''
        Decode a list of token IDs back to the original text.
        :param token_ids: List[int]: The list of token IDs to decode.
        :return: str: The decoded text.
        '''
        tokens = b''.join([self.vocab[id] for id in token_ids])
        text = tokens.decode('utf-8', errors='replace')
        logging.info(f"Decoded IDs to text: {text} successfully.")
        return text
    
    def get_stats(self, ids: List[int]) -> dict:
        '''
        Get the frequency of each adjacent token pair in the list of token IDs.
        :param ids: List[int]: The list of token IDs.
        :return: dict: A dictionary with token pairs as keys and their frequencies as values.
        '''
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def merge_pair(self, pair: Tuple[int, int], ids: List[int], idx: int) -> List[int]:
        '''
        Merge all occurrences of the given pair in the list of token IDs.
        :param pair: Tuple[int, int]: The token pair to merge.
        :param ids: List[int]: The list of token IDs.
        :param idx: int: The new token ID for the merged pair.
        :return: List[int]: The updated list of token IDs after merging.
        '''
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
                new_ids.append(idx)  # Assuming new token ID is vocab_size
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids
    
    def get_vocab(self) -> dict:
        '''
        Get the vocabulary dictionary.
        :return: dict: Vocabulary mapping token IDs to byte sequences.
        '''
        self.vocab = {i: bytes([i]) for i in range(self.single_byte_size)}
        for pair, idx in self.merges.items():
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
        return self.vocab
    
    def get_merges(self) -> dict:
        '''
        Get the merges dictionary.
        :return: dict: Merges mapping token pairs to new token IDs.
        '''
        return self.merges
    
    def save(self, vocab_path: str, merges_path: str):
        '''
        Save the vocabulary and merges to files.
        :param vocab_path: str: Path to save the vocabulary.
        :param merges_path: str: Path to save the merges.
        :return: None
        '''
        try:
            vocab_data = {str(k): v.decode('latin1') for k, v in self.get_vocab().items()}
            merges_data = {f"{k[0]},{k[1]}": v for k, v in self.get_merges().items()}
            save_json(vocab_data, vocab_path)
            save_json(merges_data, merges_path)
            logging.info(f"Vocabulary saved to {vocab_path} and merges saved to {merges_path} successfully.")
        except Exception as e:
            raise CustomException(e, sys)
    
    def load(self, vocab_path: str, merges_path: str):
        '''
        Load the vocabulary and merges from files.
        :param vocab_path: str: Path to load the vocabulary from.
        :param merges_path: str: Path to load the merges from.
        :return: None
        '''
        try:
            vocab_data = load_json(vocab_path)
            merges_data = load_json(merges_path)
            self.vocab = {int(k): v.encode('latin1') for k, v in vocab_data.items()}
            self.merges = {tuple(map(int, k.split(','))): v for k, v in merges_data.items()}
            logging.info(f"Vocabulary loaded from {vocab_path} and merges loaded from {merges_path} successfully.")
        except Exception as e:
            raise CustomException(e, sys)