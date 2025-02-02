from flask import app
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from fastapi import UploadFile, HTTPException
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from transformers import AdamW
from sklearn.metrics import accuracy_score

from app.main import extract_text_from_pdf

class LegalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }

def train_model():
    # Load dataset
    dataset = load_dataset("lex_glue", "ecthr_a")
    
    # Initialize model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained("nlpaueb/legal-bert-small-uncased", num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-small-uncased")
    
    # Preprocess data
    def preprocess_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)
    
    encoded_dataset = dataset.map(preprocess_function, batched=True)
    
    # Create dataloaders
    train_loader = DataLoader(encoded_dataset['train'], batch_size=8, shuffle=True)
    val_loader = DataLoader(encoded_dataset['validation'], batch_size=8)
    
    # Training setup
    optimizer = AdamW(model.parameters(), lr=2e-5)
    
    with mlflow.start_run():
        # Training loop
        for epoch in range(3):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['label']
                )
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                mlflow.log_metric("train_loss", loss.item())
        
        # Validation
        model.eval()
        val_predictions, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                predictions = torch.argmax(outputs.logits, dim=1)
                val_predictions.extend(predictions.cpu().numpy())
                val_labels.extend(batch['label'].cpu().numpy())
        
        accuracy = accuracy_score(val_labels, val_predictions)
        mlflow.log_metric("val_accuracy", accuracy)
        
        # Save model
        mlflow.pytorch.log_model(model, "model")
        tokenizer.save_pretrained("./app/models/legal_bert")

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model_path = "./app/models/legal_bert"
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path)

@app.post("/analyze", 
          summary="Analyze legal document",
          description="Processes a PDF document and classifies its legal clauses using a fine-tuned BERT model",
          response_description="Classification results")
async def analyze_document(file: UploadFile):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
            
        text = extract_text_from_pdf(file.file)
        if not text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            
        encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=1)
        return {"classification": predictions.tolist()}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def chunk_text(text, max_length=512):
    words = text.split()
    chunks = [' '.join(words[i:i + max_length]) for i in range(0, len(words), max_length)]
    return chunks
