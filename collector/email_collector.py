import json
import os
import re
import schedule
import time
from datetime import datetime
from pathlib import Path
from imap_tools import MailBox, AND
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer

class EmailDataCollector:
    """
    Collects emails from your inbox for SLM training
    """
    def __init__(self, email_address, app_password, output_dir="email_data"):
        self.email = email_address
        self.app_password = app_password
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.raw_emails_file = self.output_dir / "raw_emails.json"
        self.cleaned_texts_file = self.output_dir / "cleaned_texts.txt"
        self.dataset_file = self.output_dir / "training_dataset.json"
    
    def fetch_emails(self, mailbox_name="INBOX", limit=100):
        """
        Fetch emails from Gmail IMAP
        
        Args:
            mailbox_name: Mailbox to fetch from (e.g., "INBOX", "Sent Mail")
            limit: Maximum number of emails to fetch
        
        Note: For Gmail, you need to use an App Password:
        1. Go to https://myaccount.google.com/apppasswords
        2. Generate a 16-character app password
        3. Use that password instead of your regular password
        """
        print(f"[*] Connecting to Gmail...")
        
        emails_data = []
        
        try:
            with MailBox('imap.gmail.com').login(self.email, self.app_password) as mailbox:
                print(f"[✓] Connected to Gmail")
                print(f"[*] Fetching emails from {mailbox_name}...")
                
                # Fetch emails
                messages = mailbox.fetch(limit=limit, reverse=True)
                
                for msg in messages:
                    try:
                        email_dict = {
                            'subject': msg.subject,
                            'from': msg.from_,
                            'to': msg.to,
                            'date': str(msg.date),
                            'body': msg.text if msg.text else msg.html,
                            'is_html': msg.html is not None and msg.text is None
                        }
                        emails_data.append(email_dict)
                    except Exception as e:
                        print(f"[!] Error processing email: {e}")
                        continue
                
                print(f"[✓] Fetched {len(emails_data)} emails")
        
        except Exception as e:
            print(f"[!] Error connecting to Gmail: {e}")
            print(f"    Make sure you're using an App Password, not your regular password")
            return []
        
        # Save raw emails
        with open(self.raw_emails_file, 'w') as f:
            json.dump(emails_data, f, indent=2)
        print(f"[✓] Saved raw emails to {self.raw_emails_file}")
        
        return emails_data
    
    def clean_text(self, text):
        """
        Clean email text for training
        """
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '[EMAIL]', text)
        
        # Remove phone numbers
        text = re.sub(r'(\+\d{1,3}[-.\s]?)?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', '[PHONE]', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'\"-]', '', text)
        
        # Remove very short emails
        if len(text.split()) < 5:
            return ""
        
        return text
    
    def preprocess_emails(self, emails_data):
        """
        Clean and preprocess emails for training
        """
        print(f"\n[*] Preprocessing {len(emails_data)} emails...")
        
        cleaned_texts = []
        
        for idx, email in enumerate(emails_data):
            # Combine subject and body
            combined_text = f"Subject: {email['subject']} {email['body']}"
            
            # Clean
            cleaned = self.clean_text(combined_text)
            
            if cleaned:  # Only keep non-empty texts
                cleaned_texts.append(cleaned)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(emails_data)}")
        
        # Save cleaned texts
        with open(self.cleaned_texts_file, 'w') as f:
            for text in cleaned_texts:
                f.write(text + '\n')
        
        print(f"[✓] Preprocessed {len(cleaned_texts)} emails")
        print(f"[✓] Saved to {self.cleaned_texts_file}")
        
        return cleaned_texts
    
    def create_training_dataset(self, cleaned_texts, max_length=512):
        """
        Create tokenized dataset for SLM training
        """
        print(f"\n[*] Creating training dataset...")
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        dataset_items = []
        
        for idx, text in enumerate(cleaned_texts):
            # Tokenize
            encodings = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            dataset_items.append({
                'input_ids': encodings['input_ids'].squeeze().tolist(),
                'attention_mask': encodings['attention_mask'].squeeze().tolist(),
                'text': text[:200]  # Store first 200 chars for reference
            })
            
            if (idx + 1) % 100 == 0:
                print(f"  Tokenized {idx + 1}/{len(cleaned_texts)}")
        
        # Save dataset
        with open(self.dataset_file, 'w') as f:
            json.dump(dataset_items, f)
        
        print(f"[✓] Created dataset with {len(dataset_items)} samples")
        print(f"[✓] Saved to {self.dataset_file}")
        
        return dataset_items
    
    def collect_daily(self):
        """
        Schedule daily email collection
        
        Usage:
            collector = EmailDataCollector(email, password)
            collector.collect_daily()  # Runs at 9 AM every day
        """
        def job():
            print(f"\n[{datetime.now()}] Running scheduled email collection...")
            emails = self.fetch_emails(limit=50)
            if emails:
                cleaned = self.preprocess_emails(emails)
                self.create_training_dataset(cleaned)
                print(f"[✓] Daily collection completed!")
        
        # Schedule for 9 AM every day
        schedule.every().day.at("09:00").do(job)
        
        print(f"[✓] Scheduled daily email collection at 09:00")
        print(f"[*] Keep this script running to execute scheduled tasks")
        
        # Keep scheduler running
        while True:
            schedule.run_pending()
            time.sleep(60)


class EmailDataset(Dataset):
    """
    PyTorch Dataset for training SLM on email data
    """
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


# Example usage
if __name__ == "__main__":
    print("=" * 80)
    print("EMAIL DATA COLLECTOR FOR SLM TRAINING")
    print("=" * 80)
    
    print("\n[SETUP INSTRUCTIONS]")
    print("\n1. Enable 2-Step Verification on your Google Account")
    print("   Go to: https://myaccount.google.com/security")
    print("\n2. Generate App Password for Gmail")
    print("   Go to: https://myaccount.google.com/apppasswords")
    print("   Select: Mail and Windows (or other)")
    print("   Copy the 16-character password")
    print("\n3. Use the App Password in this script (NOT your regular password!)")
    
    # Your credentials
    email_address = "joneswilliamaa@gmail.com"
    app_password = "iqsr eahv ntai pncg"
    
    # Run email collection
    collector = EmailDataCollector(email_address, app_password)
    
    # One-time collection and training prep
    print("\n[*] Starting email collection from your Gmail account...")
    emails = collector.fetch_emails(limit=100)
    
    if emails:
        cleaned = collector.preprocess_emails(emails)
        dataset = collector.create_training_dataset(cleaned)
        
        # Load dataset for training
        train_dataset = EmailDataset(collector.dataset_file)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        print(f"\n[✓] Email dataset ready for training!")
    
    print("\n[QUICK START]")
    print("1. Set your email and app password above")
    print("2. Run the collection process")
    print("3. Use EmailDataset with DataLoader for training")
    
    print("\n[DAILY AUTO-COLLECTION]")
    print("To run daily collection:")
    print("  collector = EmailDataCollector(email, password)")
    print("  collector.collect_daily()  # Runs at 09:00 every day")
    
    print("\n" + "=" * 80)
    
    # Demo: Create sample dataset
    print("\n[DEMO] Creating sample training dataset...")
    
    sample_texts = [
        "Please review the quarterly report and provide your feedback by Friday",
        "The project deadline has been moved to next month due to resource constraints",
        "Thanks for the great presentation yesterday. It was very informative.",
        "I need to schedule a meeting to discuss the new feature requirements",
        "The bug fix has been deployed to production successfully",
    ]
    
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset_items = []
    
    for text in sample_texts:
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
    
    sample_dataset_file = Path("email_data/sample_dataset.json")
    sample_dataset_file.parent.mkdir(exist_ok=True)
    
    with open(sample_dataset_file, 'w') as f:
        json.dump(dataset_items, f, indent=2)
    
    print(f"[✓] Created sample dataset with {len(dataset_items)} samples")
    print(f"[✓] Saved to {sample_dataset_file}")
    
    # Load and show dataset
    demo_dataset = EmailDataset(sample_dataset_file)
    print(f"\n[✓] Successfully loaded dataset!")
    print(f"    Dataset size: {len(demo_dataset)}")
    print(f"    Sample shape: {demo_dataset[0]['input_ids'].shape}")
