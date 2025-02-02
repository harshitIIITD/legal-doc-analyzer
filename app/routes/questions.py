# Python
from fastapi import APIRouter, HTTPException, Request
from transformers import pipeline

router = APIRouter()

# Global variable to store the latest legal document text
document_context = None

# Initialize the question-answering pipeline (you can choose a different model if needed)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def set_document_context(context: str):
    global document_context
    document_context = context

@router.post("/ask")
async def ask_question(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        raise HTTPException(status_code=400, detail="No question provided")
    if not document_context:
        raise HTTPException(status_code=400, detail="No document analyzed yet")
    
    # Run the QA pipeline using the question and stored document context
    result = qa_pipeline(question=question, context=document_context)
    answer = result.get("answer", "I'm not sure about that.")
    return {"answer": answer}