import torch
import tiktoken
from model.model import GPTConfig, GPTLanguageModel

def generate_text(prompt: str, max_new_tokens: int = 50):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # Load checkpoint
    ckpt_path = "checkpoints/v4_wiki/step_001000.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # Reconstruct Config
    config_dict = checkpoint["config"]
    config = GPTConfig(**config_dict)
    
    model = GPTLanguageModel(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n--- Prompt: {prompt} ---")
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config.max_seq_len:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] # focus on last time step
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)

    generated_text = tokenizer.decode(idx[0].tolist())
    print(generated_text)

if __name__ == "__main__":
    generate_text("The history of science is")