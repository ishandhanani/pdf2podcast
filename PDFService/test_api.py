import requests
import os

PDF_SERVICE_URL = os.getenv("PDF_SERVICE_URL", "http://localhost:8000/convert")

def test_convert_pdf_endpoint():
    # Path to a sample PDF file for testing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_pdf_path = os.path.join(current_dir, "sample.pdf")

    # Ensure the sample PDF file exists
    if not os.path.exists(sample_pdf_path):
        print(f"Error: Sample PDF file not found at {sample_pdf_path}")
        return False

    # Open the PDF file and send it in the request
    with open(sample_pdf_path, "rb") as pdf_file:
        files = {"file": ("sample.pdf", pdf_file, "application/pdf")}
        response = requests.post(PDF_SERVICE_URL, files=files)

    # Run the checks
    if response.status_code != 200:
        print("Error: Request failed with status code", response.status_code)
        return False

    if not response.text:
        print("Error: Response is empty")
        return False

    if "# " not in response.text:
        print("Error: No headings found in response")
        return False

    if "- " not in response.text:
        print("Error: No list items found in response")
        return False

    if "\n\n" not in response.text:
        print("Error: No paragraph breaks found in response")
        return False

    print("Success: PDF conversion test passed!")
    print("Response content:")
    print(response.text)
    return True

def test_convert_pdf_endpoint_invalid_file():
    files = {"file": ("test.txt", b"This is not a PDF file", "text/plain")}
    response = requests.post(PDF_SERVICE_URL, files=files)

    if response.status_code == 200:
        print("Error: Invalid file was accepted")
        return False
    
    print("Success: Invalid file test passed!")
    return True

if __name__ == "__main__":
    print("Running PDF Service API tests...")
    print("\nTest 1: Valid PDF conversion")
    test_convert_pdf_endpoint()
    
    print("\nTest 2: Invalid file handling")
    test_convert_pdf_endpoint_invalid_file()
