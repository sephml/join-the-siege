from scipy.spatial.distance import cosine
import numpy as np
CONFIDENCE_THRESHOLD = 0.7
BATCH_SIZE = 16
def predict_class(text, model, categories, category_embeddings):
    if not text.strip():
        return "unknown file", np.float32(0.0)
    
    text_embedding = model.encode([text])[0]

    scores = [cosine(text_embedding, emb) for emb in category_embeddings]
    best_idx = int(min(range(len(scores)), key=lambda idx: scores[idx]))

    best_score = scores[best_idx]

    if best_score > CONFIDENCE_THRESHOLD:  # Adjustable threshold
        return "unknown file", np.float32(0.0)
    return categories[best_idx], round(1 - best_score, 3)

def predict_batch(texts, model, categories, category_embeddings):
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