from pdfminer.high_level import extract_text
from PIL import Image
import pytesseract
import filetype
from io import BytesIO

def extract_text_from_pdf(file_storage):
    try:
        pdf_bytes = file_storage.read()
        file_storage.seek(0)  # Reset after read
        return extract_text(BytesIO(pdf_bytes)).strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_image(file_stream):
    try:
        image = Image.open(file_stream)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting image text: {e}")
        return ""

def extract_text_from_file(file_storage):
    kind = filetype.guess(file_storage.read(261))  # Read a small chunk
    file_storage.seek(0)

    if kind is None:
        return ""
    
    if kind.mime == 'application/pdf':
        return extract_text_from_pdf(file_storage)
    if kind.mime.startswith('image'):
        return extract_text_from_image(file_storage)
    
    return ""
