from flask import Flask, request, jsonify
from src.preprocessing.text_extraction import extract_text_from_file
from src.models.model_loader import get_model_and_embeddings
from src.models.predict import predict_class

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

if __name__ == '__main__':
    app.run(debug=True)
