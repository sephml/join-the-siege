import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.tasks import classify_file_task, classify_batch_task
from src.models.model_loader import get_model_and_embeddings

@pytest.fixture
def model_and_embeddings():
    """Fixture to provide model and embeddings for testing"""
    return get_model_and_embeddings()

@pytest.fixture
def mock_file_content():
    """Fixture to provide mock file content"""
    return b"This is a test invoice document"

@pytest.fixture
def mock_batch_files():
    """Fixture to provide mock batch files data"""
    return [
        {"content": b"invoice for services rendered", "filename": "invoice1.pdf"},
        {"content": b"bank statement for account 12345", "filename": "statement1.pdf"},
        {"content": b"random text that doesn't match", "filename": "random.txt"}
    ]

@patch('src.tasks.extract_text_from_file')
@patch('src.tasks.predict_class')
def test_classify_file_task(mock_predict_class, mock_extract_text, mock_file_content, model_and_embeddings):
    """Test classify_file_task with mock dependencies"""
    # Setup mocks
    mock_extract_text.return_value = "This is a test invoice document"
    mock_predict_class.return_value = ("invoice", 0.95)
    
    # Call the task
    result = classify_file_task(mock_file_content, "test.pdf")
    
    # Verify the result
    assert result == {
        "filename": "test.pdf",
        "file_class": "invoice",
        "confidence": 0.95
    }
    
    # Verify mocks were called correctly
    mock_extract_text.assert_called_once_with(mock_file_content)
    mock_predict_class.assert_called_once()

@patch('src.tasks.parallel_text_extraction')
@patch('src.tasks.predict_batch')
def test_classify_batch_task(mock_predict_batch, mock_parallel_extract, mock_batch_files, model_and_embeddings):
    """Test classify_batch_task with mock dependencies"""
    # Setup mocks
    mock_parallel_extract.return_value = [
        "invoice for services rendered",
        "bank statement for account 12345",
        "random text that doesn't match"
    ]
    mock_predict_batch.return_value = [
        ("invoice", 0.95),
        ("bank_statement", 0.92),
        ("unknown file", 0.0)
    ]
    
    # Call the task
    result = classify_batch_task(mock_batch_files)
    
    # Verify the result
    assert len(result) == 3
    assert result[0] == {
        "filename": "invoice1.pdf",
        "file_class": "invoice",
        "confidence": 0.95
    }
    assert result[1] == {
        "filename": "statement1.pdf",
        "file_class": "bank_statement",
        "confidence": 0.92
    }
    assert result[2] == {
        "filename": "random.txt",
        "file_class": "unknown file",
        "confidence": 0.0
    }
    
    # Verify mocks were called correctly
    mock_parallel_extract.assert_called_once()
    mock_predict_batch.assert_called_once()

def test_classify_file_task_empty_content():
    """Test classify_file_task with empty content"""
    result = classify_file_task(b"", "empty.pdf")
    assert result == {
        "filename": "empty.pdf",
        "file_class": "unknown file",
        "confidence": 0.0
    }

def test_classify_batch_task_empty_list():
    """Test classify_batch_task with empty list"""
    result = classify_batch_task([])
    assert result == []
