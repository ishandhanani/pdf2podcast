import pytest
import requests
import os

BASE_URL = "http://localhost:8000"  # Adjust this if your server runs on a different address

def test_convert_pdf_endpoint():
    # Path to a sample PDF file for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(current_dir, "sample.pdf")

    # Ensure the sample PDF file exists
    assert os.path.exists(sample_pdf_path), f"Sample PDF file not found at {sample_pdf_path}"

    # Open the PDF file and send it in the request
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        response = requests.post(f"{BASE_URL}/convert", files=files)

    # Check if the request was successful
    assert response.status_code == 200

    # Check if the response content is not empty
    assert response.text

    # Check if the response content contains some expected markdown elements
    assert "# " in response.text  # Heading
    assert "- " in response.text  # List item
    assert "\n\n" in response.text  # Paragraph break
    print(response.text)

    # You can add more specific checks based on the content of your sample PDF

def test_convert_pdf_endpoint_invalid_file():
    # Try to upload a non-PDF file
    files = {"file": ("test.txt", b"This is not a PDF file", "text/plain")}
    response = requests.post(f"{BASE_URL}/convert", files=files)

    # Check if the request was unsuccessful
    assert response.status_code != 200

    # You might want to implement proper error handling in your API
    # and check for specific error messages here
