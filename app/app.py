import os
import sys

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import streamlit as st
from dotenv import load_dotenv

from config.config import CHAT_HISTORY_LIMIT
from pipeline.recommend_pipeline import RealEstateRecommendationPipeline

st.set_page_config(
    page_title="AI Real Estate Assistant",
    page_icon="🏡",
    layout="wide",
)
load_dotenv()

EXAMPLES = [
    "hello",
    "what is the weather today?",
    "show me apartments in Jeddah",
    "أريد فيلا للبيع في الطائف",
    "ما هي المشاريع المتاحة في عمان؟",
    "عايز شقة للإيجار في جدة؟",
    "Show me apartments for sale in Jeddah under 500,000 SAR",
    "ما هو الطقس اليوم؟",
    "السلام عليكم",
]


@st.cache_resource
def init_pipeline():
    return RealEstateRecommendationPipeline()


try:
    pipeline = init_pipeline()
except Exception as error:
    st.error(f"Could not start the pipeline: {error}")
    st.stop()

st.title("AI Real Estate Assistant")

with st.sidebar:
    st.info(
        "Note: responses may be slow because the free models have rate limits "
        "and higher latency.\n\n"
        "ملاحظة: قد تتأخر الاستجابة لأن النماذج المجانية لها حدود للطلبات "
        "وزمن استجابة أطول."
    )

    st.header("Example questions")
    st.caption(
        "These questions test all cases: guardrails, off-topic, and the main chatbot."
    )
    for example in EXAMPLES:
        if st.button(example, use_container_width=True):
            st.session_state.pending_prompt = example
            st.rerun()

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("pending_prompt", None)
        st.rerun()


def build_history(messages: list[dict]) -> list[dict]:
    history = []
    pending_question = None

    for message in messages:
        if message["role"] == "user":
            pending_question = message["content"]
        elif message["role"] == "assistant" and pending_question:
            history.append(
                {"question": pending_question, "answer": message["content"]}
            )
            pending_question = None

    return history[-CHAT_HISTORY_LIMIT:]


def render_assistant_message(message: dict) -> None:
    st.markdown(message["content"])

    if message.get("from_history"):
        st.caption("Answered from the conversation history.")

    if message.get("fallback") == "tavily":
        st.caption("AI model unavailable — answered using live web search (Tavily).")

    rewritten = message.get("rewritten_query")
    original = message.get("original_query")
    if rewritten and rewritten != original:
        st.caption(f"Search query: {rewritten}")

    filters = message.get("filters") or {}
    if filters:
        readable = ", ".join(f"{key}={value}" for key, value in filters.items())
        st.caption(f"Metadata filters applied: {readable}")

    sources = message.get("sources") or []
    if sources:
        with st.expander("Sources"):
            for source in sources:
                label = source["title"]
                if source["city"]:
                    label = f"{label} — {source['city']}"
                st.markdown(f"- **{label}** — [Open page]({source['url']})")


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

pending_prompt = st.session_state.pop("pending_prompt", None)
typed_prompt = st.chat_input("Ask about real estate properties")
prompt = typed_prompt or pending_prompt

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    history = build_history(st.session_state.messages[:-1])

    with st.chat_message("assistant"):
        with st.spinner("Searching the property pages..."):
            try:
                result = pipeline.recommend(prompt, history=history)
            except Exception as error:
                st.error(str(error))
                st.stop()

        sources = [
            {
                "title": document.metadata.get("title", "Property page"),
                "source": document.metadata.get("source", ""),
                "city": document.metadata.get("city", ""),
                "url": document.metadata.get("url", ""),
            }
            for document in result.get("sources", [])
        ]

        assistant_message = {
            "role": "assistant",
            "content": result["answer"],
            "sources": sources,
            "filters": result.get("filters") or {},
            "original_query": result.get("original_query", prompt),
            "rewritten_query": result.get("rewritten_query", prompt),
            "from_history": result.get("from_history", False),
            "fallback": result.get("fallback"),
        }

        render_assistant_message(assistant_message)
        st.session_state.messages.append(assistant_message)
