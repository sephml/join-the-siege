from scipy.spatial.distance import cosine

CONFIDENCE_THRESHOLD = 0.7

def predict_class(text, model, categories, category_embeddings):
    if not text.strip():
        return "unknown file", 0.0
    
    text_embedding = model.encode([text])[0]

    scores = [cosine(text_embedding, emb) for emb in category_embeddings]
    best_idx = int(min(range(len(scores)), key=lambda idx: scores[idx]))

    best_score = scores[best_idx]

    if best_score > CONFIDENCE_THRESHOLD:  # Adjustable threshold
        return "unknown file", 0.0
    return categories[best_idx], round(1 - best_score, 3)
