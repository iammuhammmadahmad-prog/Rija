import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
import tiktoken

class WikipediaStreamDataset(IterableDataset):
    def __init__(self, seq_len=256, lang="en", split="train"):
        super().__init__()
        self.seq_len = seq_len
        self.dataset = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split=split, streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        buffer = []
        for item in self.dataset:
            text = item["text"]
            tokens = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
            buffer.extend(tokens)
            
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[: self.seq_len + 1]
                buffer = buffer[self.seq_len :]
                
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y

def get_wiki_dataloader(batch_size=32, seq_len=256):
    ds = WikipediaStreamDataset(seq_len=seq_len)
    return DataLoader(ds, batch_size=batch_size)

if __name__ == "__main__":
    print("Testing Wikipedia streaming pipeline...")
    loader = get_wiki_dataloader(batch_size=4, seq_len=128)
    for x, y in loader:
        print(f"Batch input shape: {x.shape}, Batch target shape: {y.shape}")
        break
    print("Streaming successful!")