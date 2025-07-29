import pdfplumber
import re
import logging
from typing import List, Dict, Optional
from qa_generator import UniversalQAGenerator
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFQAExtractor:
    def __init__(self, config: Dict):
        self.config = config
        # Check the enable_qa flag in the correct location - text_extraction section
        enable_qa = config.get("text_extraction", {}).get("enable_qa", False)
        self.qa_generator = UniversalQAGenerator(config) if enable_qa else None
        
        if enable_qa:
            logger.info("QA generation is enabled")
        else:
            logger.warning("QA generation is disabled. Check text_extraction.enable_qa in config")

    def extract_text(self, pdf_path: str) -> str:
        """Extract raw text from PDF"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            logger.info(f"Extracted {len(text)} characters from {pdf_path}")
        except Exception as e:
            logger.error(f"Text extraction failed: {str(e)}")
        return text

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract Q&A pairs from a single PDF"""
        if not os.path.exists(pdf_path):
            logger.warning(f"File not found: {pdf_path}")
            return []

        text = self.extract_text(pdf_path)
        if not text:
            logger.warning(f"No text extracted from {pdf_path}")
            return []

        if self.qa_generator:
            qa_pairs = self.qa_generator.process(text)
            logger.info(f"Generated {len(qa_pairs)} Q&A pairs from {pdf_path}")
            return qa_pairs
        else:
            logger.debug("QA generator not initialized, returning empty list")
            return []