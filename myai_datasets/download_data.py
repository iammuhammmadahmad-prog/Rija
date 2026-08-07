import os
import urllib.request

def download_dataset():
    # Corrected URL (tinyshakespeare)
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_dir = "datasets/processed"
    output_file = os.path.join(output_dir, "train.txt")

    os.makedirs(output_dir, exist_ok=True)
    print("Downloading training dataset...")
    
    # Request with User-Agent header to avoid blocking
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(output_file, 'wb') as out_file:
        out_file.write(response.read())
    
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"Dataset successfully saved to {output_file} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    download_dataset()