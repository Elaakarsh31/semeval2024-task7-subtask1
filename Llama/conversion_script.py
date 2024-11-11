# conversion_script.py
import json
import torch
from pathlib import Path
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, LlamaConfig

def convert_llama_weights(model_path):
    # Load config
    with open(Path(model_path) / "params.json") as f:
        params = json.load(f)
    with open(Path(model_path) / "config.json") as f:
        config = json.load(f)
        
    # Create Llama config
    hf_config = LlamaConfig(
        hidden_size=config.get("dim", 4096),
        intermediate_size=config.get("hidden_dim", 11008),
        num_attention_heads=config.get("n_heads", 32),
        num_hidden_layers=config.get("n_layers", 32),
        rms_norm_eps=config.get("norm_eps", 1e-6),
        vocab_size=config.get("vocab_size", 32000)
    )
    
    # Load weights
    consolidated_weights = torch.load(
        Path(model_path) / "consolidated.00.pth",
        map_location="cpu"
    )
    
    # Create model with config
    model = AutoModelForCausalLM.from_config(hf_config)
    
    # Load state dict
    model.load_state_dict(consolidated_weights, strict=False)
    
    # Save in HF format
    output_dir = "./llama2-3b-hf"
    model.save_pretrained(output_dir)
    
    # Copy tokenizer files
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

if __name__ == "__main__":
    model_path = "C:\\Users\\GinMa\\.llama\\checkpoints\\Llama3.2-3B"
    output_path = convert_llama_weights(model_path)
    print(f"Model converted and saved to {output_path}")