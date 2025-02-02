import pytest
from app.main import extract_text_from_pdf

def test_extract_text_from_pdf():
    with open("tests/sample.pdf", "rb") as file:
        text = extract_text_from_pdf(file)
    assert text is not None
    assert len(text) > 0
    assert "sample text" in text
