from flask import Flask, request, jsonify
from src.celery_worker import celery
from src.tasks import classify_file_task, classify_batch_task

app = Flask(__name__)

@app.route('/classify_file', methods=['POST'])
def classify_file_route():
    """
    Classify a single file using Celery.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Start the Celery task
    task = classify_file_task.delay(file.read(), file.filename)
    
    return jsonify({"task_id": task.id}), 202

@app.route('/classify_batch', methods=['POST'])
def classify_batch_route():
    """
    Classify a batch of files using Celery.
    """
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded"}), 400

    # Prepare files data for the task
    files_data = [
        {
            'content': file.read(),
            'filename': file.filename
        }
        for file in files
    ]

    # Start the Celery task
    task = classify_batch_task.delay(files_data)
    
    return jsonify({"task_id": task.id}), 202

@app.route('/task_status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """
    Get the status of a Celery task.
    """
    task = celery.AsyncResult(task_id)
    
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is pending...'
        }
    elif task.state == 'SUCCESS':
        response = {
            'state': task.state,
            'result': task.result
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info)
        }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
