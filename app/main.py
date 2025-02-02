from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os
import mlflow
from datetime import datetime
from pydantic import BaseModel
from typing import List
from fastapi import HTTPException
from PyPDF2 import PdfReader
from app.routes.questions import router as questions_router
from app.routes.questions import set_document_context
 
 
print("Imported all libraries")
app = FastAPI()

# Mount static files and setup templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# MLFlow setup
mlflow.set_tracking_uri("http://127.0.0.1:5001")
EXPERIMENT_NAME = "Legal_Document_Analysis"
mlflow.set_experiment(EXPERIMENT_NAME)

# Model configuration
MODEL_PATH = "./app/models/legal_bert"
MODEL_NAME = "nlpaueb/legal-bert-small-uncased"
os.makedirs(MODEL_PATH, exist_ok=True)

# Load or download model
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
except:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.save_pretrained(MODEL_PATH)
    tokenizer.save_pretrained(MODEL_PATH)

class AnalysisResult(BaseModel):
    filename: str
    classification: List[int]
    text_preview: str
    confidence: float
    model_version: str
    timestamp: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
def status():
    return {"status": "healthy"}

@app.post("/analyze")
async def analyze_document(file: UploadFile):
    with mlflow.start_run():
        # Extract text
        text = extract_text_from_pdf(file.file)
        
        # Save document context for question answering
        set_document_context(text)
        
        # Model inference
        encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        outputs = model(**encoding)
        predictions = torch.argmax(outputs.logits, dim=1)
        confidence = outputs.logits.softmax(dim=1).max().item()
        
        # Log metrics and parameters
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_metric("confidence", confidence)
        mlflow.log_metric("text_length", len(text))
        
        # Save model version
        model_info = mlflow.pytorch.log_model(model, "model")
        
        # Format results
        result = AnalysisResult(
            filename=file.filename,
            classification=predictions.tolist(),
            text_preview=text[:500] + "..." if len(text) > 500 else text,
            confidence=confidence,
            model_version=model_info.model_uuid,
            timestamp=datetime.now().isoformat()
        )
        
        return JSONResponse(content=result.dict())

def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text from PDF: {str(e)}")

app.include_router(questions_router)
