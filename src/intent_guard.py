from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

from config.config import CHAT_HISTORY_LIMIT
from src.prompt_catalog import (
    GREETING_RESPONSES,
    OFF_TOPIC_RESPONSES,
    get_intent_guard_prompt,
)
from utils.json_utils import parse_json_object
from utils.llm import get_rewrite_model
from utils.logger import get_logger

logger = get_logger(__name__)

VALID_INTENTS = {"greeting", "real_estate", "off_topic"}


class IntentResult(BaseModel):
    intent: str = "real_estate"
    language: str = "en"
    response: str | None = Field(None, description="Canned reply for greeting/off-topic.")


class IntentGuard:
    def __init__(self, history_limit: int = CHAT_HISTORY_LIMIT):
        self.history_limit = history_limit
        self.model = get_rewrite_model()
        self.chain = get_intent_guard_prompt() | self.model | StrOutputParser()

    def _format_history(self, history: list[dict]) -> str:
        turns = []
        for turn in history[-self.history_limit :]:
            question = str(turn.get("question", "")).strip()
            answer = str(turn.get("answer", "")).strip()
            if question:
                turns.append(f"Question: {question}\nAnswer: {answer}")
        return "\n\n".join(turns) or "(empty)"

    def check(self, query: str, history: list[dict] | None = None) -> IntentResult:
        history = history or []

        try:
            raw = self.chain.invoke(
                {"history": self._format_history(history), "question": query}
            )
            payload = parse_json_object(raw)

            intent = str(payload.get("intent", "real_estate")).strip().lower()
            if intent not in VALID_INTENTS:
                intent = "real_estate"

            language = (
                "ar"
                if str(payload.get("language", "en")).strip().lower() == "ar"
                else "en"
            )

            text = payload.get("response")
            text = str(text).strip() if text else None

            if intent == "greeting" and not text:
                text = GREETING_RESPONSES[language]
            elif intent == "off_topic":
                text = OFF_TOPIC_RESPONSES[language]
            elif intent == "real_estate":
                text = None

            return IntentResult(intent=intent, language=language, response=text)
        except Exception as error:
            logger.warning("Intent check failed, continuing with RAG: %s", error)
            return IntentResult(intent="real_estate", language="en", response=None)
