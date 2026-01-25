import sys
import os
import torch
import yaml
import tqdm as tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logging
from utils.common import load_model, save_model, load_yaml, load_json, load_bin
from utils.exception import CustomException
from tokenizer.bpe import AlmondTokenizer
from model.gpt import AlmondGPTModel, GPTConfig, get_batch, eval_loss
# ----------------------------------


def main():
    pass