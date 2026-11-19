import argparse
import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from config.config import DATA_PATH, EMBEDDING_MODEL, PERSIST_DIR
from src.data_loader import RealEstateDataLoader
from src.data_store import StoreBuilder
from utils.custom_exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)


def main(overwrite: bool = False):
    try:
        logger.info("Starting to build the real estate RAG pipeline")

        loader = RealEstateDataLoader(DATA_PATH)
        data = loader.load_data()
        logger.info("Loaded %s records from %s", len(data), DATA_PATH)

        store_builder = StoreBuilder(
            data,
            persist_dir=PERSIST_DIR,
            embedding_model=EMBEDDING_MODEL,
        )
        store_builder.build_and_save_vectorstore(overwrite=overwrite)

        logger.info("Vector store built successfully at %s", PERSIST_DIR)
    except Exception as error:
        logger.error("Failed to build pipeline: %s", error)
        raise CustomException("Error during pipeline build", error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the Wasalt/DarGlobal vector store")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild an existing vector store")
    args = parser.parse_args()

    main(overwrite=args.overwrite)
