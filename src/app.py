from flask import Flask, request, jsonify
from src.preprocessing.text_extraction import extract_text_from_file, parallel_text_extraction
from src.models.model_loader import get_model_and_embeddings
from src.models.predict import predict_class, predict_batch

app = Flask(__name__)

model, categories, category_embeddings = get_model_and_embeddings()

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Extract text
    extracted_text = extract_text_from_file(file)

    # Predict class
    predicted_class, confidence = predict_class(extracted_text, model, categories, category_embeddings)

    return jsonify({"file_class": predicted_class, "confidence": float(confidence)}), 200

@app.route('/classify_batch', methods=['POST'])
def classify_batch_route():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    texts = parallel_text_extraction(files)
    predictions = predict_batch(texts, model, categories, category_embeddings)

    results = []
    filenames = [file.filename for file in files]

    for fname, (label, confidence) in zip(filenames, predictions):
        results.append({
            "filename": fname,
            "file_class": label,
            "confidence": float(confidence)
        })

    return jsonify(results), 200

if __name__ == '__main__':
    app.run(debug=True)
