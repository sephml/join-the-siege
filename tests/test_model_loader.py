import pytest
from src.models.model_loader import (
    model,
    CATEGORIES,
    category_embeddings,
    get_model_and_embeddings
)
from sentence_transformers import SentenceTransformer
import numpy as np

def test_model_initialization():
    """Test that the model is properly initialized as a SentenceTransformer"""
    assert isinstance(model, SentenceTransformer)
    assert model.get_sentence_embedding_dimension() == 384  # all-MiniLM-L6-v2 has 384 dimensions

def test_categories_list():
    """Test that the categories list contains the expected document types"""
    expected_categories = [
        "drivers_license",
        "bank_statement",
        "invoice",
        "passport",
        "utility_bill",
    ]
    assert CATEGORIES == expected_categories
    assert len(CATEGORIES) == 5

def test_category_embeddings():
    """Test that category embeddings are properly generated"""
    # Check that embeddings are numpy arrays
    assert isinstance(category_embeddings, np.ndarray)
    
    # Check the shape matches number of categories and embedding dimension
    assert category_embeddings.shape == (len(CATEGORIES), model.get_sentence_embedding_dimension())
    
    # Check that embeddings are normalized (length should be close to 1)
    for embedding in category_embeddings:
        assert abs(np.linalg.norm(embedding) - 1.0) < 0.1

def test_get_model_and_embeddings():
    """Test the get_model_and_embeddings function"""
    returned_model, returned_categories, returned_embeddings = get_model_and_embeddings()
    
    # Check that returned values match the module-level variables
    assert returned_model is model
    assert returned_categories is CATEGORIES
    assert np.array_equal(returned_embeddings, category_embeddings)
    
    # Check that the function returns the correct types
    assert isinstance(returned_model, SentenceTransformer)
    assert isinstance(returned_categories, list)
    assert isinstance(returned_embeddings, np.ndarray) 