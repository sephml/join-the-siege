import pytest
from io import BytesIO
from src.preprocessing.text_extraction import (
    extract_text_from_pdf,
    extract_text_from_image,
    extract_text_from_file,
    parallel_text_extraction
)
import filetype
from PIL import Image
import os

@pytest.fixture
def sample_pdf():
    """Create a sample PDF file in memory"""
    # Create a simple PDF with some text
    pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 44>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000171 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n221\n%%EOF"
    return BytesIO(pdf_content)

@pytest.fixture
def sample_image():
    """Create a sample image file in memory"""
    # Create a simple image with some text
    img = Image.new('RGB', (100, 100), color='white')
    return img

def test_extract_text_from_pdf(sample_pdf):
    """Test PDF text extraction"""
    # Test successful extraction
    text = extract_text_from_pdf(sample_pdf)
    assert isinstance(text, str)
    # Check if all characters from "Test Document" are present in any order
    assert all(char in text for char in "Test Document")

    # Test with invalid PDF
    invalid_pdf = BytesIO(b"Not a PDF")
    text = extract_text_from_pdf(invalid_pdf)
    assert text == ""

def test_extract_text_from_image(sample_image):
    """Test image text extraction"""
    # Save image to BytesIO
    img_bytes = BytesIO()
    sample_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    # Test successful extraction
    text = extract_text_from_image(img_bytes)
    assert isinstance(text, str)

    # Test with invalid image
    invalid_image = BytesIO(b"Not an image")
    text = extract_text_from_image(invalid_image)
    assert text == ""

def test_extract_text_from_file():
    """Test file type detection and text extraction"""
    # Test PDF file
    pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 44>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Test Document) Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000171 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n221\n%%EOF"
    pdf_file = BytesIO(pdf_content)
    text = extract_text_from_file(pdf_file)
    assert isinstance(text, str)
    # Check if all characters from "Test Document" are present in any order
    assert all(char in text for char in "Test Document")

    # Test image file
    img = Image.new('RGB', (100, 100), color='white')
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    text = extract_text_from_file(img_bytes)
    assert isinstance(text, str)

    # Test unsupported file type
    txt_file = BytesIO(b"Just some text")
    text = extract_text_from_file(txt_file)
    assert text == ""

def test_parallel_text_extraction():
    """Test parallel text extraction from multiple files"""
    # Create multiple sample files
    files = []
    for i in range(3):
        pdf_content = b"%PDF-1.4\n1 0 obj\n<</Type /Catalog /Pages 2 0 R>>\nendobj\n2 0 obj\n<</Type /Pages /Kids [3 0 R] /Count 1>>\nendobj\n3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> /Contents 4 0 R>>\nendobj\n4 0 obj\n<</Length 44>>\nstream\nBT\n/F1 12 Tf\n100 700 Td\n(Doc " + str(i).encode() + b") Tj\nET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\n0000000171 00000 n\ntrailer\n<</Size 5/Root 1 0 R>>\nstartxref\n221\n%%EOF"
        files.append(BytesIO(pdf_content))

    # Test parallel extraction
    results = parallel_text_extraction(files)
    
    # Verify results
    assert len(results) == 3
    for i, text in enumerate(results):
        assert isinstance(text, str)
        # Check if all characters from "Doc X" are present in any order
        assert all(char in text for char in f"Doc {i}") 