#!/usr/bin/env python3
"""
Email Data Collection Script for SLM Training
User: joneswilliamaa@gmail.com
"""

import json
import sys
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch

# Import from collector module
sys.path.insert(0, str(Path(__file__).parent))
from email_collector import EmailDataCollector, EmailDataset

def main():
    print("=" * 80)
    print("EMAIL DATA COLLECTION FOR SLM TRAINING")
    print("=" * 80)
    
    # Your Gmail credentials
    email = "joneswilliamaa@gmail.com"
    
    print(f"\n[✓] Email Address: {email}")
    
    print("\n[STEP 1] Get Your App Password")
    print("-" * 80)
    print("To collect emails from your Gmail account, you need an App Password.")
    print("\n1. Go to: https://myaccount.google.com/apppasswords")
    print("2. Sign in with your Google account")
    print("3. Make sure 2-Step Verification is ON")
    print("4. Select 'Mail' and your device type")
    print("5. Google will generate a 16-character password")
    print("6. Copy it (includes spaces like: xxxx xxxx xxxx xxxx)")
    
    # Try to get app password
    app_password = input("\nEnter your 16-character App Password (or press Enter to skip): ").strip()
    
    if app_password and len(app_password.replace(" ", "")) == 16:
        print("\n[STEP 2] Collecting Emails...")
        print("-" * 80)
        
        collector = EmailDataCollector(email, app_password, output_dir="email_data")
        
        # Fetch emails
        print(f"[*] Attempting to fetch emails from {email}...")
        emails = collector.fetch_emails(limit=100)
        
        if emails:
            print(f"\n[STEP 3] Processing Emails...")
            print("-" * 80)
            
            # Clean and preprocess
            cleaned = collector.preprocess_emails(emails)
            
            # Create training dataset
            dataset = collector.create_training_dataset(cleaned)
            
            print(f"\n[STEP 4] Dataset Ready!")
            print("-" * 80)
            print(f"[✓] Total emails collected: {len(emails)}")
            print(f"[✓] Cleaned texts: {len(cleaned)}")
            print(f"[✓] Training samples: {len(dataset)}")
            print(f"[✓] Dataset saved to: {collector.dataset_file}")
            
            print(f"\n[NEXT] Train your SLM model with this data:")
            print("  python MODEL/train_slm.py")
            
        else:
            print("\n[!] No emails collected. Using sample data instead...")
            create_sample_dataset(email)
    
    else:
        print("\n[*] No valid App Password provided.")
        print("    Creating sample dataset for testing...\n")
        create_sample_dataset(email)
    
    print("\n" + "=" * 80)


def create_sample_dataset(email):
    """Create sample dataset based on user's email style"""
    print("\n[STEP 2] Creating Sample Training Data")
    print("-" * 80)
    
    # Sample emails that mimic typical professional communication
    sample_emails = [
        "Hi, please review the project proposal and provide feedback by end of week.",
        "Meeting rescheduled to next Tuesday at 2 PM. Looking forward to your input.",
        "Thanks for the comprehensive update. The progress looks great so far.",
        "I wanted to follow up on the action items from our last discussion.",
        "Please find attached the quarterly report. Let me know if you need clarifications.",
        "Great work on the implementation. Can we schedule a follow-up meeting?",
        "The code review is complete. Some minor suggestions in the pull request.",
        "Reminder: Team standup tomorrow at 10 AM. See you there!",
        "I've completed the initial analysis. Results attached for your review.",
        "Looking forward to collaborating on this new initiative.",
    ]
    
    print(f"[*] Creating {len(sample_emails)} sample training examples...")
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    dataset_items = []
    
    for idx, text in enumerate(sample_emails):
        encodings = tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_tensors='pt'
        )
        
        dataset_items.append({
            'input_ids': encodings['input_ids'].squeeze().tolist(),
            'attention_mask': encodings['attention_mask'].squeeze().tolist(),
            'text': text
        })
    
    # Save dataset
    output_dir = Path("email_data")
    output_dir.mkdir(exist_ok=True)
    dataset_file = output_dir / "sample_dataset.json"
    
    with open(dataset_file, 'w') as f:
        json.dump(dataset_items, f, indent=2)
    
    print(f"[✓] Sample dataset created!")
    print(f"[✓] Saved to: {dataset_file}")
    print(f"[✓] Total samples: {len(dataset_items)}")
    
    # Load and verify
    dataset = EmailDataset(dataset_file)
    print(f"\n[✓] Dataset verified!")
    print(f"    Samples: {len(dataset)}")
    print(f"    Sample 1: '{sample_emails[0][:50]}...'")
    
    print(f"\n[NEXT] Train your SLM model with this sample data:")
    print("  python MODEL/train_slm.py")
    
    print(f"\n[NOTE] Once you have your App Password, collect real emails:")
    print("  python collector/collect_emails.py")
    
    return dataset_file


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        print("\n[*] Creating sample dataset instead...")
        create_sample_dataset("joneswilliamaa@gmail.com")
