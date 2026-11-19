from langchain_core.output_parsers import StrOutputParser

from src.prompt_catalog import get_real_estate_prompt
from utils.llm import get_chat_model


class RealEstateRecommender:
    def __init__(self, retriever, filter_extractor=None):
        self.retriever = retriever
        self.filter_extractor = filter_extractor
        self.llm = get_chat_model()
        self.prompt = get_real_estate_prompt()
        self.answer_chain = self.prompt | self.llm | StrOutputParser()

    def get_answer(self, query: str) -> dict:
        filters = None
        where = None
        predicate = None

        if self.filter_extractor is not None:
            filters = self.filter_extractor.extract(query)
            where = self.filter_extractor.build_where(filters)
            predicate = self.filter_extractor.build_predicate(where)

        documents = self.retriever.invoke(query, where=where, predicate=predicate)
        context = "\n\n---\n\n".join(document.page_content for document in documents)
        answer = self.answer_chain.invoke({"context": context, "question": query})

        applied = filters.model_dump(exclude_none=True) if filters is not None else {}
        return {"answer": answer, "sources": documents, "filters": applied}
