from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_finbert():
    logger.info("Downloading FinBERT model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        logger.info("✓ FinBERT downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
        return False

if __name__ == '__main__':
    download_finbert()
