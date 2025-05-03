from pdfminer.high_level import extract_text
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import pytesseract
import filetype
from io import BytesIO
from typing import List

def extract_text_from_pdf(file_storage:BytesIO) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_storage (BytesIO): The file storage object containing the PDF file.

    Returns:
        str: The extracted text from the PDF file.
    """
    try:
        pdf_bytes = file_storage.read()
        file_storage.seek(0)  # Reset after read
        return extract_text(BytesIO(pdf_bytes)).strip()
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

def extract_text_from_image(file_storage:BytesIO) -> str:
    """
    Extract text from an image file.

    Args:
        file_storage (BytesIO): The file storage object containing the image file.

    Returns:
        str: The extracted text from the image file.
    """
    try:
        image = Image.open(file_storage)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        print(f"Error extracting image text: {e}")
        return ""

def extract_text_from_file(file_storage:BytesIO) -> str:
    """
    Extract text from a file.

    Args:
        file_storage (BytesIO): The file storage object containing the file.

    Returns:
        str: The extracted text from the file.
    """
    kind = filetype.guess(file_storage.read(261))  # Read a small chunk
    file_storage.seek(0)

    if kind is None:
        return ""
    
    if kind.mime == 'application/pdf':
        return extract_text_from_pdf(file_storage)
    if kind.mime.startswith('image'):
        return extract_text_from_image(file_storage)
    
    return ""

def parallel_text_extraction(files:List[BytesIO]) -> List[str]:
    """
    Extract text from multiple files in parallel.

    Args:
        files (List[BytesIO]): A list of file storage objects containing the files to extract text from.

    Returns:
        List[str]: A list of the extracted text from each file.
    """
    with ThreadPoolExecutor() as executor:
        return list(executor.map(extract_text_from_file, files))
