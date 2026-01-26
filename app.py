import os
import sys
import asyncio
import torch
import uvicorn
from fastapi import FastAPI
from src.utils.common import load_model
from src.tokenizer.bpe import AlmondTokenizer
from src.model.gpt import AlmondGPTModel
from src.tokenizer.configs import TokenizerConfig
from src.model.configs import GPTConfig
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from src.app.schemas import GenerateRequest

# Initialize fastapi
app = FastAPI(title='AlmondGPT')

# Setup environment
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AlmondGPTModel()
tokenizer = AlmondTokenizer()

app.mount('/static', StaticFiles(directory="static"), name='static')

async def generate_chunks(prompt: str, max_new_token: int):
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device=device)
    generated = input_ids
    for _ in range(max_new_token):
        # Get logits
        logits, _ = model(generated)
        logits = logits[:, -1, :]
        
        # Get next token and concatenate with context ids
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat((generated, next_token), dim=-1)
        
        # Decode tokens
        word = tokenizer.decode([next_token.item()])
        yield f"data: {word}\n\n"
        await asyncio.sleep(0.01)

@app.post("/streaming")
async def streaming_chat(request: GenerateRequest):
    return StreamingResponse(generate_chunks(request.prompt, request.max_new_tokens), media_type='text/streaming')

@app.get("/")
async def index():
    return FileResponse("static/index.html")

# Health check
@app.get("/health")
def health_check():
    return {'status':'alive','model':'AlmondGPT'}