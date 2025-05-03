from io import BytesIO
import pytest
from src.app import app
from src.preprocessing.file_utils import allowed_file
import os
from unittest.mock import patch, MagicMock

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_celery():
    with patch('src.app.celery') as mock:
        mock.AsyncResult.return_value = MagicMock(
            state='SUCCESS',
            result={'file_class': 'invoice', 'confidence': 0.95}
        )
        yield mock

@pytest.mark.parametrize("filename, expected", [
    ("file.pdf", True),
    ("file.png", True),
    ("file.jpg", True),
    ("file.txt", False),
    ("file", False),
])
def test_allowed_file(filename, expected):
    assert allowed_file(filename) == expected

def test_no_file_in_request(client):
    response = client.post('/classify_file')
    assert response.status_code == 400

def test_no_selected_file(client):
    data = {'file': (BytesIO(b""), '')}  # Empty filename
    response = client.post('/classify_file', data=data, content_type='multipart/form-data')
    assert response.status_code == 400

@pytest.mark.parametrize("filename, expected", [
    ("invoice_1.pdf", "invoice"),
    ("invoice_2.pdf", "invoice"),
    ("invoice_3.pdf", "invoice"),
    ("drivers_license_1.jpg", "drivers_license"),
    ("drivers_license_3.jpg", "drivers_license"),
    ("drivers_licence_2.jpg", "drivers_license"),
    ("bank_statement_1.pdf", "bank_statement"),
    ("bank_statement_2.pdf", "bank_statement"),
    ("bank_statement_3.pdf", "bank_statement")
])
def test_success(client, mocker, filename, expected):
    # Mock the Celery task
    mock_task = mocker.MagicMock()
    mock_task.id = 'test-task-id'
    mocker.patch('src.app.classify_file_task.delay', return_value=mock_task)

    # Get the file path
    file_path = os.path.join('files', filename)

    # Open the file and send it directly
    with open(file_path, 'rb') as f:
        data = {'file': (f, filename)}
        response = client.post('/classify_file', data=data, content_type='multipart/form-data')

    assert response.status_code == 202
    assert response.get_json()['task_id'] == 'test-task-id'

def test_task_status(client, mocker):
    # Mock the Celery AsyncResult
    mock_result = mocker.MagicMock()
    mock_result.state = 'SUCCESS'
    mock_result.result = {'file_class': 'invoice', 'confidence': 0.95}
    mocker.patch('src.app.celery.AsyncResult', return_value=mock_result)

    response = client.get('/task_status/test-task-id')
    assert response.status_code == 200
    result = response.get_json()
    assert result['state'] == 'SUCCESS'
    assert result['result']['file_class'] == 'invoice'

def test_classify_batch(client, mocker):
    # Mock the Celery task
    mock_task = mocker.MagicMock()
    mock_task.id = 'test-batch-task-id'
    mocker.patch('src.app.classify_batch_task.delay', return_value=mock_task)

    data = {
        "files": [
            (BytesIO(b"invoice for 1000 USD to this company"), "invoice_1.pdf"),
            (BytesIO(b"bank statement for 1000 USD"), "bank_statement_2.pdf"),
            (BytesIO(b"drivers license for John Doe"), "drivers_license_1.jpg"),
        ]
    }

    response = client.post("/classify_batch", data=data, content_type="multipart/form-data")
    
    assert response.status_code == 202
    result = response.get_json()
    assert result['task_id'] == 'test-batch-task-id'
