import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import json
from pathlib import Path
from tqdm import tqdm
import sys

# Add MODEL directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "MODEL"))
sys.path.insert(0, str(Path(__file__).parent.parent / "collector"))

from slm_model import SmallLanguageModel

# Import EmailDataset directly since we know where it is
class EmailDataset:
    """Load email training dataset"""
    def __init__(self, dataset_file):
        self.dataset_file = Path(dataset_file)
        
        with open(self.dataset_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"[✓] Loaded {len(self.data)} training samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        return {
            'input_ids': torch.tensor(item['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(item['attention_mask'], dtype=torch.long),
        }

class SLMTrainer:
    """Trainer for SLM on email data"""
    
    def __init__(self, model, train_loader, val_loader=None, device='cpu'):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask)
            
            # Compute loss (shift targets by 1 for causal LM)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            loss = self.criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self):
        """Validate on validation set"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask)
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                
                loss = self.criterion(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                total_loss += loss.item()
        
        avg_loss = total_loss / len(self.val_loader)
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def train(self, num_epochs, save_dir="checkpoints"):
        """Train for multiple epochs"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n{'=' * 80}")
        print(f"TRAINING SLM ON EMAIL DATA")
        print(f"{'=' * 80}")
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"{'=' * 80}\n")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
            
            # Train
            train_loss = self.train_epoch()
            print(f"Train Loss: {train_loss:.4f}")
            
            # Validate
            if self.val_loader:
                val_loss = self.validate()
                print(f"Val Loss: {val_loss:.4f}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_dir / f"best_model.pt")
                    print(f"[✓] Saved best model (val_loss: {val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch + 1}.pt")
                print(f"[✓] Saved checkpoint at epoch {epoch + 1}")
        
        print(f"\n{'=' * 80}")
        print(f"Training completed!")
        print(f"{'=' * 80}\n")
        
        return self.train_losses, self.val_losses
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
        print(f"  Checkpoint saved to: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        print(f"[✓] Loaded checkpoint from {path}")


def generate_email(model, prompt, max_length=50, temperature=0.7, device='cpu'):
    """Generate email text using trained SLM"""
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            logits = model(input_ids)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Sample from distribution
            next_token = torch.multinomial(
                torch.softmax(next_token_logits, dim=-1), num_samples=1
            )
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we hit EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated


if __name__ == "__main__":
    print("=" * 80)
    print("SLM TRAINING ON EMAIL DATA")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Try combined dataset first, fall back to single account
    possible_paths = [
        Path(__file__).parent.parent / "DATASET" / "email_data" / "email_data" / "combined_training_dataset.json",
        Path(__file__).parent.parent / "DATASET" / "email_data" / "training_dataset.json",
        Path(__file__).parent.parent / "email_data" / "combined_training_dataset.json",
        Path(__file__).parent.parent / "email_data" / "training_dataset.json",
    ]
    
    dataset_file = None
    for path in possible_paths:
        if path.exists():
            dataset_file = path
            print(f"[✓] Found dataset at: {path}")
            break
    
    if dataset_file.exists():
        print(f"\n[*] Loading email dataset...")
        print(f"    File: {dataset_file}")
        dataset = EmailDataset(dataset_file)
        
        # Create data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2)
        
        print(f"[✓] Dataset loaded")
        print(f"    Total samples: {len(dataset)}")
        print(f"    Train samples: {len(train_dataset)}")
        print(f"    Val samples: {len(val_dataset)}")
        
        # Initialize model
        print(f"\n[*] Initializing SLM model...")
        model = SmallLanguageModel()
        print(f"[✓] Model initialized")
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Initialize trainer
        trainer = SLMTrainer(model, train_loader, val_loader, device=device)
        
        # Train
        print(f"\n[*] Starting training...")
        num_epochs = 1
        train_losses, val_losses = trainer.train(num_epochs=num_epochs)
        
        print(f"\nTraining Summary:")
        print(f"  Final Train Loss: {train_losses[-1]:.4f}")
        if val_losses:
            print(f"  Final Val Loss: {val_losses[-1]:.4f}")
        
        # Generate sample emails
        print(f"\n[*] Generating sample emails with trained model...")
        print("=" * 80)
        
        prompts = [
            "Hi, I wanted to",
            "Please review the",
            "Thanks for your",
            "Meeting scheduled for",
        ]
        
        for prompt in prompts:
            generated = generate_email(model, prompt, max_length=30, device=device)
            print(f"\nPrompt: '{prompt}'")
            print(f"Generated: '{generated}'")
        
        print(f"\n" + "=" * 80)
        print(f"[✓] Training Complete!")
        print(f"    Model checkpoints saved to: checkpoints/")
        print(f"    Best model: checkpoints/best_model.pt")
    
    else:
        print(f"\n[!] No dataset found!")
        print(f"    Please run: python collector/collect_emails.py")
        print(f"    Or: python collector/collect_multi_accounts.py")
    
    print("\n" + "=" * 80)
