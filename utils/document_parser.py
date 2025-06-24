import fitz  # PyMuPDF
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DocumentParser:
    @staticmethod
    def parse_pdf_bytes(pdf_bytes: bytes) -> str:
        """Parse PDF from bytes"""
        try:
            pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num, page in enumerate(pdf):
                page_text = page.get_text()
                if page_text.strip():
                    pages.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
            
            pdf.close()
            return "\n".join(pages).strip()
        except Exception as e:
            logger.error(f"Failed to parse PDF: {e}")
            raise
    
    @staticmethod
    def parse_text_file(content: bytes, encoding: str = 'utf-8') -> str:
        """Parse text file from bytes"""
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            try:
                return content.decode('latin-1')
            except Exception as e:
                logger.error(f"Failed to parse text file: {e}")
                raise
