# File Classifier

## Overview

This project is a file classification system that uses AI to process and categorize documents. It's designed to handle various file types (pdf, png, jpg) and can be scaled to process large volumes of documents efficiently.

## Features

- File classification based on content and filename
- Support for multiple file formats (extendable to word docs, excel, etc)
- Scalable architecture with Celery for async background processing
- Docker support for easy deployment
- Comprehensive test suite

## Prerequisites

- Docker and Docker Compose
- Python 3.10+ (for local development)
- Git

## Running with Docker

The easiest way to run the project is using Docker Compose:

1. Clone the repository:
   ```bash
   cd join-the-siege
   ```

2. Build and start the containers:
   ```bash
   docker-compose up --build
   ```

This will start three services:
- Web server (Flask) on port 8000
- Celery worker for background tasks
- Redis for message brokering

3. Access the API:
   - The API will be available at `http://localhost:8000`
   - Use the `/classify_file` endpoint to classify documents

## Local Development

If you prefer to run the project locally:

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Start Redis (required for Celery):
   ```bash
   docker run -d -p 6379:6379 redis:7-alpine
   ```

4. Run the Flask application:
   ```bash
   python -m src.app
   ```

5. In a separate terminal, start the Celery worker:
   ```bash
   celery -A src.celery_worker worker --loglevel=info
   ```

## Running Tests

The project uses pytest for testing. To run the tests:

1. Make sure you have all dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the tests:
   ```bash
   pytest -s
   ```

For more detailed test output:
   ```bash
   pytest -v
   ```

To run tests with coverage report:
   ```bash
   pytest --cov=src tests/
   ```

## API Endpoints

- `POST /classify_file`: Classify a document
  - Accepts multipart form data with a 'file' field
  - Returns classification results in JSON format

- `POST /classify_batch`: Classify a batch of documents
  - Accepts 

## Project Structure

```
.
├── src/                # Source code
├── tests/             # Test files
├── files/             # Sample files for testing
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
└── requirements.txt   # Python dependencies
```

## Improvements

All notable improvements to this project are documented in this file.

---

### Added
- Introduced ML-based classification using pretrained `SentenceTransformer` instead of filename heuristics.
- Integrated text extraction from PDFs and images using `pdfminer.six` and `pytesseract`.
- Added batch classification endpoint (`/classify_batch`) using efficient batched embedding.
- Created asynchronous endpoints powered by Celery and Redis.
- Added task polling endpoint (`/task_status/<task_id>`) for async result retrieval.
- Included unit and integration tests using `pytest`, with mocks for faster test runs.
- Implemented `Dockerfile` for containerised deployment with Gunicorn and all dependencies.
- Replaced Flask development server with Gunicorn for production readiness.
- Parallelized text extraction using `ThreadPoolExecutor` for better performance on batch jobs.
- Refactored project into modular structure: `models/`, `preprocessing/`, and `tasks.py`.
- Documented API endpoints, usage instructions, and architecture overview in `Updates.md`.

### Changed
- Removed filename-based classification logic in favor of content-based approach.
- Reorganised project layout to support scaling and testing.

