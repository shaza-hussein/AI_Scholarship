import pymupdf
import logging

logger = logging.getLogger(__name__)

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """
    Extracts raw text from PDF bytes in memory.
    """
    try:
        text = ""
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text("text") + "\n"
                
        clean_text = " ".join(text.split())
        return clean_text
    except Exception as e:
        logger.error(f"Failed to parse PDF: {e}")
        raise ValueError("Could not extract text from the provided PDF file.")