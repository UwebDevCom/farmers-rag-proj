import os
from PyPDF2 import PdfReader
from docx import Document

def read_file(file_path):
    if file_path.endswith('.pdf'):
        reader = PdfReader(file_path)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    elif file_path.endswith('.docx'):
        doc = Document(file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
    else:
        with open(file_path, 'r') as file:
            text = file.read()
    return text