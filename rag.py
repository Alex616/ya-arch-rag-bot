"""Reusable RAG assistant for the Telegram bot.

Encapsulates embedding → Chroma retrieval → prompt building → YandexGPT
(LLM called through the OpenAI-compatible API endpoint) inside a single
class that can be imported by the bot handler.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from few_shot_examples import FEW_SHOT_EXAMPLES

# Load environment variables from .env file, if present.
load_dotenv()

logger = logging.getLogger(__name__)

# These defaults mirror the values used in build_index.py so the embedding
# model and the vector DB collection are consistent.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_PERSIST_DIR = Path("chroma.db")
DEFAULT_COLLECTION_NAME = "knowledge_base"


def _default_k() -> int:
    raw = os.getenv("K_RETRIEVER", "5")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(
            f"Environment variable K_RETRIEVER must be an integer, got {raw!r}"
        ) from exc


DEFAULT_K = _default_k()

DEFAULT_YANDEX_BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 1500


def _build_prompt_template(few_shot_examples: str | None = None) -> str:
    """Build the "stuff" chain prompt with optional few-shot examples.

    {context} is filled by RetrievalQA with the concatenated retrieved
    chunks; {question} is the user query.
    """
    if few_shot_examples:
        examples_block = (
            "\n\nНиже приведены примеры того, как отвечать на основе "
            "фрагментов базы знаний:\n\n" + few_shot_examples.strip() + "\n\n"
        )
    else:
        examples_block = "\n\n"

    return (
        "Ты — корпоративный помощник на основе внутренней базы знаний.\n"
        "Ответь на вопрос сотрудника, опираясь ТОЛЬКО на приведённые ниже фрагменты.\n"
        "Если фрагментов недостаточно для ответа, честно скажи, что не знаешь ответа.\n"
        "Не придумывай факты и не используй внешние знания."
        + examples_block
        + "Фрагменты базы знаний:\n"
        "{context}\n\n"
        "Вопрос: {question}\n\n"
        "Ответ:"
    )


class RAGAssistant:
    """RAG assistant backed by ChromaDB and YandexGPT OpenAI-compatible API.

    Typical Telegram-bot usage (create **one** instance at startup):

        from rag import RAGAssistant

        assistant = RAGAssistant()

        @dp.message()
        async def handle(message: Message) -> None:
            result = assistant.ask(message.text)
            await message.answer(result["answer"])
    """

    @staticmethod
    def _parse_float_env(
        var_name: str, default: float, override: float | None = None
    ) -> float:
        """Parse an optional float from environment, falling back to default."""
        value = override if override is not None else os.getenv(var_name)
        if value is None or value == "":
            return default
        try:
            return float(value)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {var_name} must be a number, got {value!r}"
            ) from exc

    @staticmethod
    def _parse_int_env(var_name: str, default: int, override: int | None = None) -> int:
        """Parse an optional int from environment, falling back to default."""
        value = override if override is not None else os.getenv(var_name)
        if value is None or value == "":
            return default
        try:
            return int(value)  # type: ignore[arg-type]
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {var_name} must be an integer, got {value!r}"
            ) from exc

    def __init__(
        self,
        persist_dir: str | Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        k: int = DEFAULT_K,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        few_shot_examples: str | None = FEW_SHOT_EXAMPLES,
    ) -> None:
        self._few_shot_examples = few_shot_examples
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.k = k

        if not self.persist_dir.exists():
            raise FileNotFoundError(
                f"Chroma persist directory not found: {self.persist_dir.absolute()}"
            )

        self._api_key = api_key or os.getenv("YANDEX_API_KEY")
        self._base_url = base_url or os.getenv(
            "YANDEX_BASE_URL", DEFAULT_YANDEX_BASE_URL
        )
        self._model_name = model_name or os.getenv("YANDEX_MODEL", "yandexgpt-lite")
        self._temperature = self._parse_float_env(
            "YANDEX_TEMPERATURE", DEFAULT_TEMPERATURE, override=temperature
        )
        self._max_tokens = self._parse_int_env(
            "YANDEX_MAX_TOKENS", DEFAULT_MAX_TOKENS, override=max_tokens
        )

        if not self._api_key:
            raise ValueError(
                "Yandex API key is required. Set the YANDEX_API_KEY environment "
                "variable or pass api_key=... to RAGAssistant."
            )

        logger.info("Loading embedding model: %s", self.embedding_model)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Loading Chroma vector store from %s", self.persist_dir)
        self.vectorstore = Chroma(
            persist_directory=str(self.persist_dir),
            embedding_function=self.embeddings,
            collection_name=self.collection_name,
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k},
        )

        logger.info("Initializing YandexGPT model: %s", self._model_name)
        self.llm = ChatOpenAI(
            model=self._model_name,
            api_key=SecretStr(self._api_key),
            base_url=self._base_url,
            temperature=self._temperature,
            max_tokens=self._max_tokens,  # type: ignore[call-arg]
        )

        prompt = PromptTemplate(
            template=_build_prompt_template(self._few_shot_examples),
            input_variables=["context", "question"],
        )

        # Ready-made LangChain retrieval chain.
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.retriever,
            chain_type="stuff",
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )

    def ask(self, query: str) -> dict:
        """Answer a user query.

        Returns a dict with the generated answer and the retrieved sources:

            {
                "query": "...",
                "answer": "...",
                "sources": [
                    {"source": "article_005.txt", "content": "..."},
                    ...
                ],
            }
        """
        if not query or not query.strip():
            raise ValueError("Query text must not be empty.")

        query = query.strip()
        result = self.qa_chain.invoke({"query": query})

        answer = (result.get("result") or "").strip()
        sources = [
            {
                "source": doc.metadata.get("source", "unknown"),
                "content": doc.page_content.strip(),
            }
            for doc in result.get("source_documents", [])
        ]

        return {"query": query, "answer": answer, "sources": sources}


def main() -> None:
    """CLI sanity check: run a single query."""
    import sys

    default_query = "Кто такой Мясник из Беловежа и чем он занимается?"
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else default_query

    assistant = RAGAssistant()
    response = assistant.ask(query)

    print(f"Запрос: {response['query']}\n")
    print("=" * 60)
    print(response["answer"])
    print("=" * 60)
    print("\nИсточники:")
    for i, src in enumerate(response["sources"], 1):
        preview = (
            (src["content"][:300] + "…")
            if len(src["content"]) > 300
            else src["content"]
        )
        print(f"{i}. {src['source']}: {preview.replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
