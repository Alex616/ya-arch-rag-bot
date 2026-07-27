import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Конфигурация
KNOWLEDGE_DIR = Path("knowledge_base")
PERSIST_DIR = Path("chroma.db")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 128


def read_documents(directory: Path) -> list[dict]:
    """Считывает все .txt-файлы из директории с базой знаний."""
    docs = []
    for path in sorted(directory.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs


def main() -> None:
    total_start = time.perf_counter()

    print("=" * 60)
    print("Построение Chroma-индекса для knowledge_base")
    print("=" * 60)

    # 1. Загрузка документов
    print("\n[1/4] Загрузка документов...")
    raw_docs = read_documents(KNOWLEDGE_DIR)
    total_chars = sum(len(doc["text"]) for doc in raw_docs)
    print(f"        Файлов загружено: {len(raw_docs)}")
    print(f"        Всего символов: {total_chars}")

    # 2. Инициализация локальной модели эмбеддингов
    print("\n[2/4] Инициализация модели эмбеддингов...")
    model_start = time.perf_counter()
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    model_time = time.perf_counter() - model_start
    print(f"        Модель: {EMBEDDING_MODEL}")
    print(f"        Модель готова за {model_time:.2f} с")

    # 3. Разбиение документов на чанки
    print("\n[3/4] Разбиение документов на чанки...")
    split_start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    for doc in raw_docs:
        for chunk_text in splitter.split_text(doc["text"]):
            chunk_text = chunk_text.strip()
            if len(chunk_text) >= 50:
                chunks.append({"source": doc["source"], "text": chunk_text})
    split_time = time.perf_counter() - split_start
    print(f"        Размер чанка: {CHUNK_SIZE}, перекрытие: {CHUNK_OVERLAP}")
    print(f"        Всего чанков: {len(chunks)}")
    print(f"        Чанкинг завершён за {split_time:.2f} с")

    # 4. Построение и сохранение Chroma-индекса
    print("\n[4/4] Построение индекса Chroma...")
    index_start = time.perf_counter()
    if PERSIST_DIR.exists():
        try:
            shutil.rmtree(PERSIST_DIR)
            print(f"        Очищен существующий индекс: {PERSIST_DIR}")
        except OSError as exc:
            print(f"        Предупреждение: не удалось очистить индекс: {exc}")

    Chroma.from_texts(
        texts=[chunk["text"] for chunk in chunks],
        embedding=embeddings,
        metadatas=[{"source": chunk["source"]} for chunk in chunks],
        persist_directory=str(PERSIST_DIR),
        collection_name="knowledge_base",
    )
    index_time = time.perf_counter() - index_start
    print(f"        Индекс построен и сохранён за {index_time:.2f} с")
    print(f"        Директория индекса: {PERSIST_DIR.resolve()}")

    # Итог
    total_time = time.perf_counter() - total_start
    print("\n" + "=" * 60)
    print("Итог")
    print("=" * 60)
    print(f"Модель эмбеддингов: {EMBEDDING_MODEL}")
    print(f"База знаний       : {KNOWLEDGE_DIR} ({len(raw_docs)} файла)")
    print(f"Чанков в индексе  : {len(chunks)}")
    print(f"Инициализация     : {model_time:.2f} с")
    print(f"Чанкинг           : {split_time:.2f} с")
    print(f"Построение индекса: {index_time:.2f} с")
    print(f"Общее время       : {total_time:.2f} с")
    print("=" * 60)


if __name__ == "__main__":
    main()
