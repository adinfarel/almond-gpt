import sys
import os
import torch
import tqdm as tqdm
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logging
from utils.common import load_model, save_model, load_yaml
from utils.exception import CustomException
from dataclasses import dataclass
from model.configs import GPTConfig
# ----------------------------------

def get_batch(data, batch_size, block_size):
    '''
    Generate a batch of data for training.
    :param data: List[int]: The tokenized dataset.
    :param batch_size: int: The size of the batch.
    :param block_size: int: The context length.
    :return: Tuple[torch.Tensor, torch.Tensor]: Input and target tensors.
    '''
    # Randomly select starting indices for each sequence in the batch
    ix = torch.randint(len(data) - block_size, (batch_size,)) # starting indices
    # Create input and target tensors
    x = torch.stack([torch.tensor(data[i:i+block_size]) for i in ix]) # input sequences
    y = torch.stack([torch.tensor(data[i+1:i+block_size+1]) for i in ix]) # target sequences
    return x, y

# Estimate loss function
@torch.no_grad()
def eval_loss(model, data, config: GPTConfig):
    '''
    Estimate the loss on the evaluation dataset.
    :param model: GPTModel: The GPT model.
    :param data: List[int]: The tokenized dataset.
    :param config: GPTConfig: The configuration object.
    :return: float: The estimated loss.
    '''
    model.eval()
    losses = []
    for _ in range(config.eval_iters):
        xb, yb = get_batch(data, config.batch_size, config.block_size)
        xb, yb = xb.to(config.device), yb.to(config.device)
        logits, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

class Head(nn.Module):
    '''
    A single attention head.
    '''
    def __init__(self, head_size, n_embd, dropout, block_size):
        super().__init__()
        # Linear layers for key, query, value projections
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # Masking to ensure causality
        self.register_buffer(
            "tril", # lower triangular matrix for masking
            torch.tril(torch.ones(block_size, block_size)) # lower triangular matrix
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B, T, C = x.shape # batch size, time steps, embedding size
        key = self.key(x) # (B, T, head_size)
        query = self.query(x) # (B, T, head_size)
        # Compute attention scores ("affinities")
        attn = query @ key.transpose(-2, -1) * (self.head_size ** -0.5) # Scaled dot-product attention and normalization (B, T, T)
        # Apply causal mask to ensure each position can only attend to previous positions
        wei = attn.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # Softmax to get probabilities
        wei = self.dropout(wei)
        # Compute the weighted sum of values
        value = self.value(x) # (B, T, head_size)
        out = wei @ value # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention mechanism.
    '''
    def __init__(self, num_heads, head_size, n_embd, dropout, block_size):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embd, dropout, block_size) for _ in range(num_heads)]
        )
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Concatenate outputs from all heads
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    '''
    Feed-forward neural network.
    '''
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    '''
    A single Transformer block.
    '''
    def __init__(self, n_embd, n_heads, dropout, block_size):
        super().__init__()
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size, n_embd, dropout, block_size) # i.e. multi-head self-attention with 64 head size
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Residual connection around self-attention and pre-norm to LayerNorm
        x = x + self.ffwd(self.ln2(x)) # Residual connection around feed-forward and pre-norm to LayerNorm
        return x

class AlmondGPTModel(nn.Module):
    '''
    A simple GPT model implementation.
    '''
    def __init__(self):
        super().__init__()
        self.config = GPTConfig.from_yaml("configs/models_config.yaml")
        self.token_emb = nn.Embedding(self.config.vocab_size, self.config.n_embd)
        self.pos_emb = nn.Embedding(self.config.block_size, self.config.n_embd)
        self.blocks = nn.Sequential(
            *[Block(self.config.n_embd, self.config.n_heads, self.config.dropout, self.config.block_size) for _ in range(self.config.n_layers)]
        )
        self.ln_f = nn.LayerNorm(self.config.n_embd)
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)
    

    def forward(self, x, targets=None):
        B, T = x.shape # batch size, time steps
        # token embeddings and positional embeddings
        token_embeddings = self.token_emb(x) # (B, T, n_embd)
        position_embeddings = self.pos_emb(torch.arange(T, device=self.config.device)) # (T, n_embd)
        x = token_embeddings + position_embeddings # (B, T, n_embd)
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(-1, C)
            loss = F.cross_entropy(logits, targets.view(-1))
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        '''
        Generate new tokens given a context.
        :param idx: torch.Tensor: The input context tensor of shape (B, T).
        :param max_new_tokens: int: The number of new tokens to generate.
        :return: torch.Tensor: The generated token tensor of shape (B, T + max_new_tokens).
        '''
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.config.block_size:]
            # Get the logits for the current context
            logits, _ = self.forward(idx_cond) # (B, T, vocab_size)
            # Focus on the last time step
            logits = logits[:, -1, :] # (B, vocab_size)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, vocab_size)
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append the sampled token to the sequence
            idx = torch.cat((idx, next_token), dim=1) # (B, T+1)
        return idx
    
# ----------------------------------
if __name__ == "__main__":
    # Example usage
    config = GPTConfig.from_yaml("configs/models_config.yaml")
    model = AlmondGPTModel().to(config.device)
    logging.info("GPT Model initialized successfully.")
        

