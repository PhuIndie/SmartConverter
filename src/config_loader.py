import yaml
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config() -> Dict[str, Any]:
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
            logger.info("Config loaded successfully.")
            return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def load_pdf_list() -> list:
    try:
        with open("config/pdf_sources.yaml", "r") as f:
            pdfs = yaml.safe_load(f).get("pdfs", [])
            logger.info(f"Loaded {len(pdfs)} PDFs.")
            return pdfs
    except Exception as e:
        logger.error(f"Failed to load PDF list: {str(e)}")
        raise