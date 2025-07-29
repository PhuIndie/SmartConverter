import os
import logging
from typing import List, Dict
from config_loader import load_config, load_pdf_list
from pdf_extractor import PDFQAExtractor
from json_builder import save_qa_pairs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def process_all_pdfs(pdf_list: List[Dict], config: Dict) -> List[Dict]:
    """Process all PDFs and collect Q&A pairs"""
    extractor = PDFQAExtractor(config)
    all_qa_pairs = []

    for pdf_info in pdf_list:
        pdf_path = os.path.join(config["input"]["pdf_dir"], pdf_info["path"])
        logger.info(f"Processing: {pdf_info['name']}")

        qa_pairs = extractor.process_pdf(pdf_path)
        all_qa_pairs.extend(qa_pairs)

    return all_qa_pairs


def main():
    try:
        logger.info("Starting Q&A extraction pipeline")

        # Load configuration
        config = load_config()
        pdf_list = load_pdf_list()

        # Process all PDFs
        qa_pairs = process_all_pdfs(pdf_list, config)

        # Save results
        output_path = save_qa_pairs(qa_pairs, config["output"]["json_dir"])
        logger.info(f"Saved {len(qa_pairs)} Q&A pairs to {output_path}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()