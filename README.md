# Legal Document Analyzer

## Overview

The Legal Document Analyzer is a FastAPI application designed to analyze legal documents using a fine-tuned BERT model. The application provides endpoints for uploading PDF files and analyzing their content. It also includes a web interface for interacting with the application.

## Main Features

- Upload and analyze legal documents (PDF)
- Extract text from PDF files
- Classify legal clauses using a fine-tuned BERT model
- Track experiments and log metrics using MLflow
- Web interface for uploading documents and viewing results
- API endpoints for document analysis and question answering

## Setup

### Dependencies

The project requires the following dependencies:

- Python 3.8+
- FastAPI
- Uvicorn
- Transformers
- Torch
- PyPDF2
- MLflow
- Docker
- DVC
- Datasets
- Python-Multipart

### Installation

1. Clone the repository:

```bash
git clone https://github.com/harshitIIITD/legal-doc-analyzer.git
cd legal-doc-analyzer
```

2. Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

4. Set up MLflow:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

## Running the Application

1. Start the FastAPI application:

```bash
uvicorn app.main:app --reload
```

2. Open your browser and navigate to `http://127.0.0.1:8000` to access the web interface.

## Using the API

### Analyze Document

Endpoint: `POST /analyze`

Upload a PDF document to analyze its content.

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@path/to/document.pdf"
```

#### Response

```json
{
  "filename": "document.pdf",
  "classification": [0, 1, 0, 1],
  "text_preview": "Extracted text preview...",
  "confidence": 0.95,
  "model_version": "1234567890abcdef",
  "timestamp": "2023-01-01T12:00:00"
}
```

### Ask Question

Endpoint: `POST /ask`

Ask a question about the analyzed document.

#### Request

```bash
curl -X POST "http://127.0.0.1:8000/ask" -H "Content-Type: application/json" -d '{"question": "What is the main clause?"}'
```

#### Response

```json
{
  "answer": "The main clause is..."
}
```

## Architecture

![Architecture](docs/architecture.png)

## Data Flow

![Data Flow](docs/data_flow.png)

## Key Components

- `app/main.py`: Main FastAPI application
- `app/routes/questions.py`: Question answering endpoint
- `app/models/legal_bert`: Fine-tuned BERT model
- `app/templates/index.html`: Web interface template
- `tests`: Test cases for the application
- `data/dvc.yaml`: Data version control configuration
