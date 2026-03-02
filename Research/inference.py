# inference.py
import torch
import time
import os
import json
from dataclasses import dataclass
from transformers import GPT2TokenizerFast
from model import TitanLLM

# 1. Configuration (Must match training)
@dataclass
class LLMConfig:
    vocab_size: int = 50257
    max_seq_len: int = 512
    n_layers: int = 6       # Updated to match Part 2 final config
    n_heads: int = 8
    d_model: int = 512
    dropout: float = 0.0    # No dropout for inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TitanInferenceEngine:
    def __init__(self, model_path, config=None):
        self.cfg = config if config else LLMConfig()
        self.model = TitanLLM(self.cfg).to(DEVICE)
        
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            state_dict = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(state_dict)
        else:
            print("⚠️ Weights not found! Initializing random model.")
            
        self.model.eval()
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        
        # --- STATIC CACHE SETUP ---
        print("🧠 Allocating Static KV-Cache...")
        self.static_cache = []
        for _ in range(self.cfg.n_layers):
            k = torch.zeros(1, self.cfg.n_heads, self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads, device=DEVICE)
            v = torch.zeros(1, self.cfg.n_heads, self.cfg.max_seq_len, self.cfg.d_model // self.cfg.n_heads, device=DEVICE)
            self.static_cache.append((k, v))
            
        # --- STATIC TENSORS FOR GRAPH ---
        self.static_token = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
        self.static_pos = torch.zeros((1,), dtype=torch.long, device=DEVICE)
        
        self.cuda_graph = None
        self.static_logits = None
        
        # Try capturing graph
        if torch.cuda.is_available():
            self._capture_graph()

    def _capture_graph(self):
        print("📸 Capturing CUDA Graph...")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            # Warmup
            for _ in range(3):
                self.model(self.static_token, input_pos=self.static_pos, kv_cache=self.static_cache)
            
            # Capture
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                self.static_logits, _ = self.model(self.static_token, input_pos=self.static_pos, kv_cache=self.static_cache)
                
        torch.cuda.current_stream().wait_stream(s)
        print("✅ Graph Captured.")

    def generate(self, prompt, max_new_tokens=100, temperature=0.8):
        # 1. Prefill (Standard Forward Pass)
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        B, T = tokens.shape
        
        # Reset Cache
        for k, v in self.static_cache: k.zero_(); v.zero_()
        
        start_time = time.time()
        
        with torch.no_grad():
            # Process prompt
            input_pos = torch.arange(0, T, device=DEVICE)
            logits, _ = self.model(tokens, input_pos=input_pos, kv_cache=self.static_cache)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            
            generated = [next_token.item()]
            
            # 2. Decode (Graph Replay)
            cur_pos = T
            for _ in range(max_new_tokens):
                # Copy inputs to static buffers
                self.static_token.copy_(next_token)
                self.static_pos.copy_(torch.tensor([cur_pos], device=DEVICE))
                
                # Replay Graph
                if self.cuda_graph:
                    self.cuda_graph.replay()
                    logits = self.static_logits
                else:
                    logits, _ = self.model(self.static_token, input_pos=self.static_pos, kv_cache=self.static_cache)
                
                # Sample
                logits = logits[:, -1, :] / temperature
                probs = torch.nn.functional.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated.append(next_token.item())
                cur_pos += 1
                if cur_pos >= self.cfg.max_seq_len: break

        torch.cuda.synchronize()
        dt = time.time() - start_time
        print(f"🚀 Speed: {max_new_tokens / dt:.2f} tokens/sec")
        
        return self.tokenizer.decode(tokens[0].tolist() + generated)

if __name__ == "__main__":
    # Example usage
    engine = TitanInferenceEngine("titan_tinystories.pth")
    story = engine.generate("Once upon a time", max_new_tokens=200)
    print("\n" + "="*40)
    print(story)
    print("="*40)
