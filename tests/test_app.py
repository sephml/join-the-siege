from io import BytesIO
import pytest
from src.app import app
from src.preprocessing.file_utils import allowed_file
import os

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

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
    # Get the file path
    file_path = os.path.join('files', filename)

    # Open the file and send it directly
    with open(file_path, 'rb') as f:
        data = {'file': (f, filename)}
        response = client.post('/classify_file', data=data, content_type='multipart/form-data')

    assert response.status_code == 200
    assert response.get_json()['file_class'] == expected

def test_classify_batch(client, mocker):
    # Simulate text extraction for all files
    mocker.patch(
        "src.preprocessing.text_extraction.extract_text_from_file",
        side_effect=lambda f: f"mock text from {f.filename}"
    )

    # Simulate ML predictions
    mock_predictions = [
        ("invoice", 0.93),
        ("bank_statement", 0.91),
        ("drivers_license", 0.88)
    ]
    mocker.patch(
        "src.models.predict.predict_batch",
        return_value=mock_predictions
    )

    data = {
        "files": [
            (BytesIO(b"invoice for 1000 USD to this company"), "invoice_1.pdf"),
            (BytesIO(b"bank statement for 1000 USD"), "bank_statement_2.pdf"),
            (BytesIO(b"drivers license for John Doe"), "drivers_license_1.jpg"),
        ]
    }

    response = client.post("/classify_batch", data=data, content_type="multipart/form-data")
    
    assert response.status_code == 200
    result = response.get_json()

    assert isinstance(result, list)
    assert len(result) == 3

    for item, (expected_class, expected_conf) in zip(result, mock_predictions):
        assert item["file_class"] == expected_class
        assert item["filename"] in ["invoice_1.pdf", "bank_statement_2.pdf", "drivers_license_1.jpg"]
