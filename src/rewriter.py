from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from config.config import CHAT_HISTORY_LIMIT
from src.prompt_catalog import get_query_rewrite_prompt
from utils.json_utils import parse_json_object
from utils.llm import get_rewrite_model
from utils.logger import get_logger

logger = get_logger(__name__)


class RewriteResult(BaseModel):
    rewritten_query: str
    use_rag: bool = True
    answer: str | None = Field(
        None, description="Answer taken from the chat history when RAG is not needed."
    )


class QueryRewriter:
    def __init__(self, history_limit: int = CHAT_HISTORY_LIMIT):
        self.history_limit = history_limit
        self.model = get_rewrite_model()
        self.chain = get_query_rewrite_prompt() | self.model | StrOutputParser()

    def _format_history(self, history: list[dict]) -> str:
        turns = []
        for turn in history[-self.history_limit :]:
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if not question:
                continue
            turns.append(f"Question: {question}\nAnswer: {answer}")
        return "\n\n".join(turns) or "(empty)"

    def rewrite(self, query: str, history: list[dict] | None = None) -> RewriteResult:
        history = history or []
        if not history:
            return RewriteResult(rewritten_query=query)

        try:
            raw = self.chain.invoke(
                {"history": self._format_history(history), "question": query}
            )
            payload = parse_json_object(raw)

            rewritten = str(payload.get("rewritten_query") or query).strip() or query
            use_rag = bool(payload.get("use_rag", True))
            answer = payload.get("answer")
            answer = str(answer).strip() if answer else None

            if not use_rag and not answer:
                use_rag = True

            return RewriteResult(
                rewritten_query=query if not use_rag else rewritten,
                use_rag=use_rag,
                answer=None if use_rag else answer,
            )
        except Exception as error:
            logger.warning("Query rewrite failed, using original question: %s", error)
            return RewriteResult(rewritten_query=query)
