# app/utils/pdf_processor.py

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_file: bytes) -> str:
    """Extrai texto de um arquivo PDF"""
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text
