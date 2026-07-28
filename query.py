"""CLI example: ask a question and get an answer from the RAG assistant.

Uses the reusable `RAGAssistant` class from `rag.py`, making it trivial to
migrate the same flow into a Telegram bot handler.
"""

import argparse
import sys

DEFAULT_QUERY = "Кто такой Мясник из Беловежа и чем он занимается?"


def main() -> None:
    # Local import keeps the script runnable as a standalone CLI example.
    # In a Telegram bot import RAGAssistant at module level once.
    from rag import RAGAssistant  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser(
        description="RAG-запрос: поиск по индексу + ответ YandexGPT"
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Текстовый запрос пользователя (если не указан — используется запрос по умолчанию)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Количество чанков, которые передаются в контекст LLM (по умолчанию 5)",
    )
    args = parser.parse_args()

    query_text = " ".join(args.query) if args.query else DEFAULT_QUERY

    # Создаём один экземпляр ассистента. В Telegram-боте его стоит создавать
    # один раз при старте приложения, чтобы не перезагружать эмбеддинг-модель.
    assistant = RAGAssistant(k=args.k)
    response = assistant.ask(query_text)

    print(f"Запрос: {response['query']}\n")
    print("=" * 60)
    print(response["answer"])
    print("=" * 60)
    print("\nИспользованные чанки:")
    for i, src in enumerate(response["sources"], 1):
        content = src["content"]
        preview = (content[:300] + "…") if len(content) > 300 else content
        print(f"{i}. {src['source']}: {preview.replace(chr(10), ' ')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nПрервано пользователем", file=sys.stderr)
        sys.exit(130)
    except ValueError as exc:
        print(f"Ошибка конфигурации: {exc}", file=sys.stderr)
        sys.exit(1)
