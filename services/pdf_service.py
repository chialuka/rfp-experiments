from typing import Dict
from langchain_core.tools import tool
from pathlib import Path
from reducto import Reducto
import os

@tool
def parse_pdf(filename: str) -> Dict:
    """Parse a PDF document using Reducto and extract the full text."""
    try:
        reducto_api_key = os.environ.get("REDUCTO_API_KEY")
        if not reducto_api_key:
            return {"error": "REDUCTO_API_KEY environment variable not set"}

        pdf_path = Path(filename)
        if not pdf_path.exists():
            return {"error": f"File not found: {filename}"}

        client = Reducto(api_key=reducto_api_key)
        upload = client.upload(file=pdf_path)
        parse_result = client.parse.run(document_url=upload)
        document_text = "\n".join([r.content for r in parse_result.result.chunks])

        return {
            "document_text": document_text,
            "document_filename": filename,
            "chunk_count": len(parse_result.result.chunks),
        }
    except Exception as e:
        return {"error": f"Error parsing PDF: {str(e)}"}


def get_pdfs_from_storage() -> Dict:
    """Get a single PDF file from the file_storage directory."""
    storage_dir = "file_storage"
    if not os.path.exists(storage_dir):
        print(f"❌ Directory '{storage_dir}' not found. Creating it...")
        os.makedirs(storage_dir)
        return None

    pdf_files = [
        os.path.join(storage_dir, f) for f in os.listdir(storage_dir) if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"❌ No PDF files found in '{storage_dir}' directory")
        return None

    if len(pdf_files) > 1:
        print(f"⚠️ Multiple PDFs found. Using only: {os.path.basename(pdf_files[0])}")

    selected_pdf = pdf_files[0]
    print(f"\n📚 Processing PDF: {os.path.basename(selected_pdf)}")
    return parse_pdf(selected_pdf) 
