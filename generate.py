import torch
import os
from src.model.gpt import AlmondGPTModel
from src.tokenizer.bpe import AlmondTokenizer
from src.utils.common import load_model
from src.model.configs import GPTConfig
from src.tokenizer.configs import TokenizerConfig

def generate_test():
    '''
    Inference models 
    '''
    model_config = GPTConfig.from_yaml("configs/models_config.yaml")
    tokenizer_config = TokenizerConfig.from_yaml("configs/tokenizer_config.yaml")
    
    device = "cpu" 
    
    tokenizer = AlmondTokenizer()
    tokenizer.load(tokenizer_config.vocab_path, tokenizer_config.merges_path)
    
    model = AlmondGPTModel().to(device)
    model = load_model(model, "models/checkpoints/gpt_model.pt", device=device)
    model.eval() 
    
    prompt = "User: Why can camels survive for long without water?" 
    print(f"\nPrompt: {prompt}")
    
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    
    print("Generating...")
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=50)[0].tolist()
    
    generated_text = tokenizer.decode(output_ids)
    
    print("-" * 30)
    print(f"Result:\n{generated_text}")
    print("-" * 30)

if __name__ == "__main__":
    generate_test()