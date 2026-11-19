from config.config import (
    DATA_PATH,
    DENSE_TOP_K,
    EMBEDDING_MODEL,
    GROQ_API_KEY,
    HYBRID_WEIGHTS,
    OPENROUTER_API_KEY,
    PERSIST_DIR,
    SPARSE_CANDIDATES,
    SPARSE_TOP_K,
)
from src.data_loader import RealEstateDataLoader
from src.data_store import StoreBuilder
from src.hybrid_retriever import FilteredHybridRetriever
from src.intent_guard import IntentGuard
from src.query_filters import MetadataFilterExtractor
from src.recommender import RealEstateRecommender
from src.rewriter import QueryRewriter
from src.search_fallback import TavilyFallback
from utils.custom_exception import CustomException
from utils.logger import get_logger

logger = get_logger(__name__)


class RealEstateRecommendationPipeline:
    def __init__(self, persist_dir: str = PERSIST_DIR, data_path: str = DATA_PATH):
        try:
            logger.info("Initializing real estate recommendation pipeline")

            if not OPENROUTER_API_KEY:
                raise ValueError(
                    "OPENROUTER_API_KEY is not set. Add it to your .env file."
                )
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

            data = RealEstateDataLoader(data_path).load_data()
            store_builder = StoreBuilder(
                data,
                persist_dir=persist_dir,
                embedding_model=EMBEDDING_MODEL,
            )

            sparse_retriever = store_builder.build_sparse_retriever()
            vectorstore = store_builder.load_vector_store()

            hybrid_retriever = FilteredHybridRetriever(
                vectorstore=vectorstore,
                sparse_retriever=sparse_retriever,
                weights=HYBRID_WEIGHTS,
                dense_k=DENSE_TOP_K,
                sparse_k=SPARSE_TOP_K,
                sparse_candidates=SPARSE_CANDIDATES,
                top_k=DENSE_TOP_K,
            )

            filter_extractor = MetadataFilterExtractor(data=data)
            self.intent_guard = IntentGuard()
            self.rewriter = QueryRewriter()
            self.tavily_fallback = TavilyFallback()

            self.recommender = RealEstateRecommender(
                retriever=hybrid_retriever,
                filter_extractor=filter_extractor,
            )

            logger.info("Pipeline initialized successfully")
        except Exception as error:
            logger.error("Failed to initialize pipeline: %s", error)
            raise CustomException("Error during pipeline initialization", error)

    def recommend(self, query: str, history: list[dict] | None = None) -> dict:
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty")

            query = query.strip()
            logger.info("Received query: %s", query)

            intent = self.intent_guard.check(query, history)
            logger.info("Detected intent: %s (%s)", intent.intent, intent.language)

            if intent.intent in {"greeting", "off_topic"}:
                return {
                    "answer": intent.response,
                    "sources": [],
                    "filters": {},
                    "original_query": query,
                    "rewritten_query": query,
                    "from_history": False,
                    "intent": intent.intent,
                    "language": intent.language,
                }

            rewrite = self.rewriter.rewrite(query, history)
            logger.info(
                "Rewritten query: %s (use_rag=%s)", rewrite.rewritten_query, rewrite.use_rag
            )

            if not rewrite.use_rag and rewrite.answer:
                return {
                    "answer": rewrite.answer,
                    "sources": [],
                    "filters": {},
                    "original_query": query,
                    "rewritten_query": rewrite.rewritten_query,
                    "from_history": True,
                    "intent": intent.intent,
                    "language": intent.language,
                }

            try:
                result = self.recommender.get_answer(rewrite.rewritten_query)
            except Exception as error:
                logger.warning(
                    "Model answer failed, falling back to Tavily search: %s", error
                )
                result = self.tavily_fallback.build_result(rewrite.rewritten_query)

            result["original_query"] = query
            result["rewritten_query"] = rewrite.rewritten_query
            result["from_history"] = False
            result["intent"] = intent.intent
            result["language"] = intent.language

            logger.info("Answer generated successfully")
            return result
        except Exception as error:
            logger.error("Failed to generate answer: %s", error)
            raise CustomException("Error during answer generation", error)
