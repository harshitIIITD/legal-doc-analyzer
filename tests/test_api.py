import pytest
from fastapi.testclient import TestClient
from app.main import app, extract_text_from_pdf, set_document_context
from app.routes.questions import qa_pipeline

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert b"Legal Document Analyzer" in response.content

def test_status():
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_analyze_document():
    with open("tests/sample.pdf", "rb") as file:
        response = client.post("/analyze", files={"file": ("sample.pdf", file, "application/pdf")})
    assert response.status_code == 200
    result = response.json()
    assert "filename" in result
    assert "classification" in result
    assert "text_preview" in result
    assert "confidence" in result
    assert "model_version" in result
    assert "timestamp" in result

def test_mlflow_setup():
    with open("tests/sample.pdf", "rb") as file:
        response = client.post("/analyze", files={"file": ("sample.pdf", file, "application/pdf")})
    assert response.status_code == 200
    result = response.json()
    assert "confidence" in result
    assert result["confidence"] > 0

def test_questions_router_integration():
    set_document_context("This is a sample legal document text.")
    response = client.post("/ask", json={"question": "What is this document about?"})
    assert response.status_code == 200
    result = response.json()
    assert "answer" in result
    assert result["answer"] != "I'm not sure about that."
