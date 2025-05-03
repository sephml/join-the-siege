from scipy.spatial.distance import cosine
import numpy as np
from sentence_transformers import SentenceTransformer

CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE = 16

def predict_class(text: str, model: SentenceTransformer, categories: list[str], category_embeddings: list[np.ndarray]) -> tuple[str, np.float32]:
    """
    Predict the class of a given text using a pre-trained model and category embeddings.
    
    Args:
        text (str): The text to predict the class of.
        model: The pre-trained model to use for prediction.
        categories: The list of categories to predict from.
        category_embeddings: The list of category embeddings.

    Returns:
        tuple[str, np.float32]: A tuple containing the predicted class and the confidence score.
    """
    if not text.strip():
        return "unknown file", np.float32(0.0)
    
    text_embedding = model.encode([text])[0]

    scores = [cosine(text_embedding, emb) for emb in category_embeddings]
    best_idx = int(min(range(len(scores)), key=lambda idx: scores[idx]))

    best_score = scores[best_idx]

    if best_score > CONFIDENCE_THRESHOLD:  # Adjustable threshold
        return "unknown file", np.float32(0.0)
    return categories[best_idx], round(1 - best_score, 3)

def predict_batch(texts: list[str], model: SentenceTransformer, categories: list[str], category_embeddings: list[np.ndarray]) -> list[tuple[str, np.float32]]:
    """
    Predict the class of a list of texts using a pre-trained model and category embeddings.
    
    Args:
        texts (list[str]): The list of texts to predict the class of.
        model: The pre-trained model to use for prediction.
        categories: The list of categories to predict from.
        category_embeddings: The list of category embeddings.

    Returns:
        list[tuple[str, np.float32]]: A list of tuples containing the predicted class and the confidence score.
    """
    embeddings = model.encode(texts, batch_size=BATCH_SIZE)  # Vectorize all texts
    predictions = []

    for emb in embeddings:
        scores = [cosine(emb, cat_emb) for cat_emb in category_embeddings]
        best_idx = int(min(range(len(scores)), key=lambda idx: scores[idx]))

        best_score = scores[best_idx]

        if best_score > CONFIDENCE_THRESHOLD:
            label = "unknown file"
            confidence = np.float32(0.0)
        else:
            label = categories[best_idx]
            confidence = round(1 - best_score, 3)
        predictions.append((label, confidence))

    return predictions