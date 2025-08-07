import os
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document

def extract_text_from_pdf(pdf_path):
    try:
        return extract_pdf_text(pdf_path)
    except Exception as e:
        print(f"Error extracting PDF {pdf_path}: {e}")
        return ""

def extract_text_from_docx(docx_path):
    try:
        doc = Document(docx_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error extracting DOCX {docx_path}: {e}")
        return ""

def extract_resume_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        print(f"Unsupported file type: {file_path}")
        return ""