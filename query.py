import sys

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Конфигурация должна совпадать с build_index.py
PERSIST_DIR = "chroma.db"
COLLECTION_NAME = "knowledge_base"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Поисковый запрос по умолчанию (на русском, как и база знаний)
DEFAULT_QUERY = "Кто такой Мясник из Беловежа и чем он занимается?"


def main() -> None:
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUERY

    print(f"Запрос: {query}\n")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )

    results = vectorstore.similarity_search(query, k=5)

    print(f"Найдено релевантных чанков: {len(results)}\n")
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        content = doc.page_content.strip()
        preview = content if len(content) <= 500 else content[:500].rstrip() + "…"
        print(f"--- Результат {i} (источник: {source}) ---")
        print(preview)
        print()


if __name__ == "__main__":
    main()
