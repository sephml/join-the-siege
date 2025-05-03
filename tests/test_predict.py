import pytest
import numpy as np
from src.models.predict import predict_class, predict_batch, CONFIDENCE_THRESHOLD
from src.models.model_loader import get_model_and_embeddings

@pytest.fixture
def model_and_embeddings():
    """Fixture to provide model and embeddings for testing"""
    return get_model_and_embeddings()

def test_predict_class_empty_text(model_and_embeddings):
    """Test prediction with empty text"""
    model, categories, category_embeddings = model_and_embeddings
    result = predict_class("", model, categories, category_embeddings)
    assert result == ("unknown file", 0.0)

def test_predict_class_whitespace_text(model_and_embeddings):
    """Test prediction with whitespace-only text"""
    model, categories, category_embeddings = model_and_embeddings
    result = predict_class("   \n\t  ", model, categories, category_embeddings)
    assert result == ("unknown file", 0.0)

def test_predict_class_known_category(model_and_embeddings):
    """Test prediction with text that should match a known category"""
    model, categories, category_embeddings = model_and_embeddings
    
    # Test with text that should match "invoice"
    result = predict_class("invoice for services rendered", model, categories, category_embeddings)
    assert result[0] == "invoice"
    assert isinstance(result[1], np.float32)
    assert 0 <= result[1] <= 1

    # Test with text that should match "bank_statement"
    result = predict_class("bank statement for account 12345", model, categories, category_embeddings)
    assert result[0] == "bank_statement"
    assert isinstance(result[1], np.float32)
    assert 0 <= result[1] <= 1

def test_predict_class_unknown_category(model_and_embeddings):
    """Test prediction with text that should not match any category"""
    model, categories, category_embeddings = model_and_embeddings
    
    # Test with random text that shouldn't match any category
    result = predict_class("random text that doesn't match any category", model, categories, category_embeddings)
    assert result[0] == "unknown file"
    assert result[1] == 0.0

def test_predict_batch_empty_list(model_and_embeddings):
    """Test batch prediction with empty list"""
    model, categories, category_embeddings = model_and_embeddings
    result = predict_batch([], model, categories, category_embeddings)
    assert result == []

def test_predict_batch_single_item(model_and_embeddings):
    """Test batch prediction with single item"""
    model, categories, category_embeddings = model_and_embeddings
    texts = ["invoice for services rendered"]
    result = predict_batch(texts, model, categories, category_embeddings)
    
    assert len(result) == 1
    assert result[0][0] == "invoice"
    assert isinstance(result[0][1], np.float32)
    assert 0 <= result[0][1] <= 1

def test_predict_batch_multiple_items(model_and_embeddings):
    """Test batch prediction with multiple items"""
    model, categories, category_embeddings = model_and_embeddings
    texts = [
        "invoice for services rendered",
        "bank statement for account 12345",
        "drivers license for John Doe",
        "random text that doesn't match"
    ]
    result = predict_batch(texts, model, categories, category_embeddings)
    
    assert len(result) == len(texts)
    for pred, conf in result:
        assert isinstance(pred, str)
        assert isinstance(conf, np.float32)
        assert 0 <= conf <= 1
        assert pred in categories or pred == "unknown file"

def test_predict_batch_confidence_threshold(model_and_embeddings):
    """Test that predictions below confidence threshold are marked as unknown"""
    model, categories, category_embeddings = model_and_embeddings
    
    # Create a mock embedding that would result in low confidence
    mock_embedding = np.random.rand(384)  # 384 is the dimension of all-MiniLM-L6-v2
    mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)  # Normalize
    
    # Mock the model.encode method to return our mock embedding
    model.encode = lambda x, **kwargs: [mock_embedding]
    
    result = predict_batch(["random text"], model, categories, category_embeddings)
    assert result[0][0] == "unknown file"
    assert result[0][1] == 0.0 