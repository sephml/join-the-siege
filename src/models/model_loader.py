from sentence_transformers import SentenceTransformer
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load a small, general-purpose embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Possible classes
CATEGORIES = [
    "drivers_license",
    "bank_statement",
    "invoice",
    "passport",
    "utility_bill",
]

# Embed class labels
category_embeddings = model.encode(CATEGORIES)

def get_model_and_embeddings() -> tuple[SentenceTransformer, list[str], list[np.ndarray]]:
    """
    Get the pre-trained model and category embeddings.

    Returns:
        tuple[SentenceTransformer, list[str], list[np.ndarray]]: A tuple containing the pre-trained model, the list of categories, and the list of category embeddings.
    """
    return model, CATEGORIES, category_embeddings
