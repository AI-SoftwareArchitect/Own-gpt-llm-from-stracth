import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Hyperparameters optimized for RTX 3050 laptop with 4GB VRAM
BATCH_SIZE = 12  # Reduced batch size for 4GB VRAM
BLOCK_SIZE = 64  # Smaller context window to fit in memory
EMBED_DIM = 192  # Reduced embedding dimension
NUM_HEADS = 6   # Fewer attention heads
NUM_LAYERS = 4  # Fewer transformer layers
LR = 5e-4      # Slightly increased learning rate for faster convergence
SAVE_INTERVAL = 100  # Save checkpoint every N batches
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_EPOCHS = 10

# Data Loading
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size):
        self.tokenizer = tokenizer
        self.examples = []
        
        for text in texts:
            tokens = self.tokenizer.encode(text)
            for i in range(0, len(tokens) - block_size, block_size // 2):  # 50% overlap for better context
                if i + block_size <= len(tokens):
                    self.examples.append(tokens[i:i+block_size])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = torch.tensor(self.examples[idx][:-1], dtype=torch.long)
        y = torch.tensor(self.examples[idx][1:], dtype=torch.long)
        return x.to(DEVICE), y.to(DEVICE)

# Simple Tokenizer
class CharTokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse_vocab = {}
        
    def train(self, texts):
        chars = sorted(list(set(''.join(texts))))
        self.vocab = {ch:i+2 for i,ch in enumerate(chars)}
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.inverse_vocab = {v:k for k,v in self.vocab.items()}
        
    def save(self, path):
        torch.save({
            'vocab': self.vocab,
            'inverse_vocab': self.inverse_vocab
        }, path)
        
    def load(self, path):
        checkpoint = torch.load(path)
        self.vocab = checkpoint['vocab']
        self.inverse_vocab = checkpoint['inverse_vocab']
    
    def encode(self, text):
        return [self.vocab.get(c, 1) for c in text]
    
    def decode(self, tokens):
        return ''.join([self.inverse_vocab.get(t, '') for t in tokens])

# Model Architecture
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4*embed_dim),
            nn.GELU(),
            nn.Linear(4*embed_dim, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size=BLOCK_SIZE):
        super().__init__()
        self.block_size = block_size
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(block_size, embed_dim)
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(0, T, device=DEVICE)
        tok_emb = self.token_embed(x)
        pos_emb = self.pos_embed(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.head(x)
        return logits

# Load text data from file
def load_text_file(file_path):
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        return [f.read()]

# Training
def train_model(continue_training=False):
    # Check if we can continue training
    start_epoch = 0
    start_batch = 0
    checkpoint_path = 'gpt_checkpoint.pth'
    tokenizer_path = 'tokenizer.pth'
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset("roneneldan/TinyStories")
    texts = dataset['train']['text'] + dataset['validation']['text']
    
    # Check for custom training data
    custom_data_path = 'new_train_data.txt'
    custom_texts = load_text_file(custom_data_path)
    if custom_texts:
        print(f"Found custom training data at {custom_data_path}. Adding to training set.")
        texts = texts + custom_texts
    
    # Initialize or load tokenizer
    tokenizer = CharTokenizer()
    if continue_training and os.path.exists(tokenizer_path):
        print("Loading existing tokenizer...")
        tokenizer.load(tokenizer_path)
    else:
        print("Training new tokenizer...")
        tokenizer.train(texts)
        tokenizer.save(tokenizer_path)
    
    vocab_size = len(tokenizer.vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create dataloader
    print("Creating dataset...")
    train_dataset = TextDataset(texts, tokenizer, BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize or load model
    model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    if continue_training and os.path.exists(checkpoint_path):
        print("Loading checkpoint to continue training...")
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint.get('batch', 0)
        print(f"Continuing from epoch {start_epoch}, batch {start_batch}")
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop
    model.train()
    total_batches = len(train_loader)
    
    print(f"Training on {'GPU' if DEVICE == 'cuda' else 'CPU'}")
    print(f"Batch size: {BATCH_SIZE}, Block size: {BLOCK_SIZE}")
    print(f"Model dimensions: Embed={EMBED_DIM}, Heads={NUM_HEADS}, Layers={NUM_LAYERS}")
    
    try:
        for epoch in range(start_epoch, MAX_EPOCHS):
            total_loss = 0
            batch_start = start_batch if epoch == start_epoch else 0
            
            pbar = tqdm(enumerate(train_loader), total=total_batches, desc=f"Epoch {epoch+1} [0%]")
            
            for i, (x, y) in pbar:
                # Skip batches we've already processed in case of continuing training
                if i < batch_start:
                    continue
                    
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                total_loss += loss.item()
                
                # Update progress bar
                percent_complete = (i+1)/total_batches*100
                pbar.set_description(f"Epoch {epoch+1} [{percent_complete:.1f}%] Loss: {loss.item():.4f}")
                
                # Save checkpoint periodically
                if (i + 1) % SAVE_INTERVAL == 0:
                    print(f"Saving checkpoint at epoch {epoch+1}, batch {i+1}...")
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'batch': i+1,
                    }, checkpoint_path)
                    
                    # Also save to numbered checkpoint for safety
                    torch.save({
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'epoch': epoch,
                        'batch': i+1,
                    }, f'checkpoints/checkpoint_e{epoch+1}_b{i+1}.pth')
            
            # Reset start_batch after first epoch when continuing
            start_batch = 0
            
            print(f"Epoch {epoch+1} Average Loss: {total_loss/total_batches:.4f}")
            
            # Save checkpoint after each epoch
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch + 1,
                'batch': 0,
            }, checkpoint_path)
            
            # Also save final model for easy loading during inference
            torch.save(model.state_dict(), 'gpt_model.pth')
            
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch,
            'batch': i+1,
        }, checkpoint_path)
        print("Checkpoint saved. You can continue training later.")
        return
        
    # Save final model
    torch.save(model.state_dict(), 'gpt_model.pth')
    print("Training complete! Model saved.")

# Text Generation
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.8):
    model.eval()
    tokens = tokenizer.encode(prompt)
    input_tokens = tokens[-BLOCK_SIZE:] if len(tokens) > BLOCK_SIZE else tokens
    
    # Pad if necessary
    if len(input_tokens) < BLOCK_SIZE:
        input_tokens = [0] * (BLOCK_SIZE - len(input_tokens)) + input_tokens
    
    generated = tokens.copy()
    
    for _ in range(max_length):
        x = torch.tensor(input_tokens[-BLOCK_SIZE:], dtype=torch.long, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        
        # Get probabilities for next token
        probs = torch.nn.functional.softmax(logits[0, -1, :] / temperature, dim=-1)
        
        # Sample from distribution
        next_token = torch.multinomial(probs, 1).item()
        
        # Add to generated sequence
        generated.append(next_token)
        input_tokens.append(next_token)
        
        # Stop if we generate end of text marker (if implemented)
        # if next_token == end_token_id:
        #     break
    
    return tokenizer.decode(generated)

# Memory usage estimation
def estimate_memory_usage():
    vocab_size = 256  # Approximate for char-level encoding
    
    # Model parameters
    token_embed_params = vocab_size * EMBED_DIM
    pos_embed_params = BLOCK_SIZE * EMBED_DIM
    
    # Each transformer block
    attn_params = 4 * EMBED_DIM * EMBED_DIM  # Q, K, V, and output projections
    ff_params = EMBED_DIM * (4 * EMBED_DIM) + (4 * EMBED_DIM) * EMBED_DIM
    block_params = attn_params + ff_params
    
    # Total parameters
    total_params = token_embed_params + pos_embed_params + (NUM_LAYERS * block_params) + EMBED_DIM * vocab_size
    
    # Activations (rough estimate)
    batch_activation_size = BATCH_SIZE * BLOCK_SIZE * EMBED_DIM * 4  # 4 bytes per float
    
    # Convert to MB
    param_memory_mb = total_params * 4 / (1024 * 1024)
    activation_memory_mb = batch_activation_size / (1024 * 1024)
    optimizer_memory_mb = param_memory_mb * 2  # Adam uses ~2x params for moment estimates
    
    total_memory_mb = param_memory_mb + activation_memory_mb + optimizer_memory_mb
    
    return {
        'parameters_mb': param_memory_mb,
        'activations_mb': activation_memory_mb,
        'optimizer_mb': optimizer_memory_mb,
        'total_memory_mb': total_memory_mb
    }

# CLI Interface
def main():
    model = None
    tokenizer = None
    
    while True:
        print("\n--- LLM CLI ---")
        print("1. Train New Model")
        print("2. Continue Training")
        print("3. Run Model")
        print("4. Memory Usage Estimation")
        print("5. Exit")
        choice = input("Seçiminiz: ")
        
        if choice == '1':
            train_model(continue_training=False)
        elif choice == '2':
            train_model(continue_training=True)
        elif choice == '3':
            if not (os.path.exists('gpt_model.pth') and os.path.exists('tokenizer.pth')):
                print("Önce modeli eğitin!")
                continue
                
            # Load model
            print("Loading model and tokenizer...")
            tokenizer = CharTokenizer()
            tokenizer.load('tokenizer.pth')
            vocab_size = len(tokenizer.vocab)
            
            model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS, BLOCK_SIZE).to(DEVICE)
            model.load_state_dict(torch.load('gpt_model.pth', map_location=DEVICE))
            
            prompt = input("Prompt girin: ")
            temp = float(input("Sıcaklık değeri (0.1-2.0 arası, önerilen: 0.8): ") or "0.8")
            length = int(input("Maksimum üretilecek token sayısı (önerilen: 100): ") or "100")
            
            print("\nMetin üretiliyor...")
            output = generate_text(model, tokenizer, prompt, max_length=length, temperature=temp)
            print("\nÜretilen Metin:")
            print(output)
        elif choice == '4':
            mem_est = estimate_memory_usage()
            print("\nTahmini Bellek Kullanımı:")
            print(f"Model Parametreleri: {mem_est['parameters_mb']:.2f} MB")
            print(f"Aktivasyonlar: {mem_est['activations_mb']:.2f} MB")
            print(f"Optimizer: {mem_est['optimizer_mb']:.2f} MB")
            print(f"Toplam: {mem_est['total_memory_mb']:.2f} MB")
            
            vram_mb = 4 * 1024  # 4GB VRAM
            remaining = vram_mb - mem_est['total_memory_mb']
            
            print(f"\nRTX 3050 4GB VRAM: {vram_mb} MB")
            print(f"Tahmini Kalan VRAM: {remaining:.2f} MB")
            
            if remaining < 500:
                print("\nUYARI: VRAM sınırına yaklaşıyorsunuz. Parametreleri düşürmeyi düşünün.")
            else:
                print("\nVRAM kullanımı uygun görünüyor.")
        elif choice == '5':
            print("Çıkış yapılıyor...")
            break
        else:
            print("Geçersiz seçim!")

if __name__ == "__main__":
    main()