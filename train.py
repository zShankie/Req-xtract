import nltk
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.model_selection import train_test_split

# Download required NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')

# Text Preprocessing
def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(str(text))
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(lemmatized_tokens)

# Custom Dataset Class
class RequirementDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Load Data & Train Model
def train_model(excel_file, model_output='fine_tuned_distilbert', epochs=3, batch_size=8):
    df = pd.read_excel(excel_file)
    if 'Requirement Text' not in df.columns or 'Type' not in df.columns:
        raise ValueError("Excel file must contain 'Requirement Text' and 'Type' columns.")
    
    df['Processed Text'] = df['Requirement Text'].apply(preprocess_text)
    df['Label'] = df['Type'].apply(lambda x: 0 if x == 'FR' else 1)
    
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['Processed Text'].tolist(), df['Label'].tolist(), test_size=0.2, random_state=42
    )
    
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_dataset = RequirementDataset(train_texts, train_labels, tokenizer)
    val_dataset = RequirementDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)

    # Define Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    loss_fn = nn.CrossEntropyLoss()

    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {correct/total:.4f}")

    # Save Model & Tokenizer
    model.save_pretrained(model_output)
    tokenizer.save_pretrained(model_output)
    print(f"Model fine-tuned and saved to {model_output}")

# Example Usage
# train_model('data.xlsx')
