#!/usr/bin/env python3
"""
Multi-Account Email Data Collection
Collects from multiple Gmail accounts for SLM training
"""

import json
from pathlib import Path
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent))
from email_collector import EmailDataCollector, EmailDataset

def collect_from_multiple_accounts():
    """Collect emails from multiple Gmail accounts"""
    
    print("=" * 80)
    print("MULTI-ACCOUNT EMAIL DATA COLLECTION")
    print("=" * 80)
    
    # Define multiple accounts
    accounts = [
        {
            "email": "joneswilliamaa@gmail.com",
            "password": "iqsr eahv ntai pncg"
        },
        {
            "email": "joneswilliam1011@gmail.com",
            "password": "pxam slaf hhfz ggmc"
        }
    ]
    
    all_emails = []
    all_cleaned_texts = []
    
    # Collect from each account
    for idx, account in enumerate(accounts, 1):
        print(f"\n[{idx}/{len(accounts)}] Collecting from: {account['email']}")
        print("-" * 80)
        
        try:
            collector = EmailDataCollector(
                account['email'],
                account['password'],
                output_dir="email_data"
            )
            
            # Fetch emails
            print(f"[*] Fetching emails...")
            emails = collector.fetch_emails(limit=100)
            
            if emails:
                print(f"[✓] Collected {len(emails)} emails")
                all_emails.extend(emails)
                
                # Preprocess
                print(f"[*] Preprocessing emails...")
                cleaned = collector.preprocess_emails(emails)
                print(f"[✓] Cleaned {len(cleaned)} emails")
                all_cleaned_texts.extend(cleaned)
            else:
                print(f"[!] No emails collected from {account['email']}")
        
        except Exception as e:
            print(f"[!] Error collecting from {account['email']}: {e}")
            continue
    
    # Combine and create unified dataset
    if all_cleaned_texts:
        print(f"\n[STEP 3] Creating Unified Dataset")
        print("=" * 80)
        print(f"[*] Total emails collected: {len(all_emails)}")
        print(f"[*] Total cleaned texts: {len(all_cleaned_texts)}")
        
        # Create combined dataset
        print(f"[*] Creating training dataset...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset_items = []
        for idx, text in enumerate(all_cleaned_texts):
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
                'text': text[:200]
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Tokenized {idx + 1}/{len(all_cleaned_texts)}")
        
        # Save combined dataset
        output_dir = Path("email_data")
        output_dir.mkdir(exist_ok=True, parents=True)
        dataset_file = output_dir / "combined_training_dataset.json"
        
        with open(dataset_file, 'w') as f:
            json.dump(dataset_items, f)
        
        print(f"\n[✓] Combined Dataset Created!")
        print(f"    Total samples: {len(dataset_items)}")
        print(f"    Saved to: {dataset_file}")
        
        # Save account info
        account_info_file = output_dir / "account_info.json"
        with open(account_info_file, 'w') as f:
            json.dump({
                "accounts": [acc["email"] for acc in accounts],
                "total_emails": len(all_emails),
                "total_samples": len(dataset_items)
            }, f, indent=2)
        
        print(f"    Account info: {account_info_file}")
        
        # Load and verify
        print(f"\n[*] Verifying dataset...")
        dataset = EmailDataset(dataset_file)
        print(f"[✓] Dataset verified!")
        print(f"    Samples: {len(dataset)}")
        print(f"    Sample tensor shape: {dataset[0]['input_ids'].shape}")
        
        print(f"\n[NEXT STEPS]")
        print("-" * 80)
        print(f"Train your SLM with combined email data:")
        print(f"  python MODEL/train_slm_multi.py")
        
        return dataset_file
    
    else:
        print(f"\n[!] No emails collected from any account!")
        return None


if __name__ == "__main__":
    try:
        dataset_file = collect_from_multiple_accounts()
        
        if dataset_file:
            print(f"\n" + "=" * 80)
            print(f"[✓] EMAIL COLLECTION COMPLETE!")
            print(f"=" * 80)
        else:
            print(f"\n[!] Collection failed. Please check your app passwords.")
    
    except KeyboardInterrupt:
        print("\n\n[!] Cancelled by user.")
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
