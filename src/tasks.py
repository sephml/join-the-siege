from celery import shared_task
from src.preprocessing.text_extraction import extract_text_from_file, parallel_text_extraction
from src.models.model_loader import get_model_and_embeddings
from src.models.predict import predict_class, predict_batch

model, categories, category_embeddings = get_model_and_embeddings()

@shared_task(name='src.tasks.classify_file_task')
def classify_file_task(file_content, filename):
    """
    Celery task for classifying a single file.
    """
    # Extract text
    extracted_text = extract_text_from_file(file_content)
    
    # Predict class
    predicted_class, confidence = predict_class(extracted_text, model, categories, category_embeddings)
    
    return {
        "filename": filename,
        "file_class": predicted_class,
        "confidence": float(confidence)
    }

@shared_task(name='src.tasks.classify_batch_task')
def classify_batch_task(files_data):
    """
    Celery task for classifying a batch of files.
    """
    files = [file_data['content'] for file_data in files_data]
    filenames = [file_data['filename'] for file_data in files_data]
    
    # Extract text in parallel
    texts = parallel_text_extraction(files)
    
    # Predict classes
    predictions = predict_batch(texts, model, categories, category_embeddings)
    
    results = []
    for fname, (label, confidence) in zip(filenames, predictions):
        results.append({
            "filename": fname,
            "file_class": label,
            "confidence": float(confidence)
        })
    
    return results
