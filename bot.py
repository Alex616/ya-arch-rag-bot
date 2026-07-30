"""Telegram-бот, отвечающий на вопросы с помощью RAG-ассистента.

Создаёт один экземпляр `RAGAssistant` при старте и переиспользует его
для всех входящих сообщений.
"""

import asyncio
import logging
import os
import sys

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import Message
from dotenv import load_dotenv

from rag import RAGAssistant

# Загружаем переменные окружения до проверки TELEGRAM_BOT_TOKEN.
load_dotenv()

_telegram_token = os.getenv("TELEGRAM_BOT_TOKEN")
if not _telegram_token:
    raise ValueError("Отсутствует переменная окружения TELEGRAM_BOT_TOKEN. "
                     "Добавьте её в .env файл или задайте в окружении.")

# Типизированная константа токена (str) для корректной проверки типов.
TELEGRAM_BOT_TOKEN: str = _telegram_token

# Настройка логирования: INFO в консоль.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Единственный экземпляр RAG-ассистента; инициализируется в main().
rag_assistant: RAGAssistant | None = None

# Диспетчер aiogram 3.x.
dp = Dispatcher()


@dp.message(Command("start", "help"))
async def cmd_start(message: Message) -> None:
    """Обработчик команд /start и /help."""
    await message.answer(
        "Привет! Я помощник по базе знаний.\n\n"
        "Задайте мне вопрос — я найду релевантные фрагменты и сформулирую "
        "ответ на основе внутренней документации.")


def split_telegram_text(text: str, max_length: int = 4096) -> list[str]:
    """Разбить длинный текст на части, укладывающиеся в лимит Telegram."""
    if len(text) <= max_length:
        return [text]

    parts: list[str] = []
    while text:
        # Берём максимально возможный кусок.
        chunk = text[:max_length]
        # Пытаемся обрезать по последнему переводу строки, чтобы не ломать
        # форматирование.
        newline_pos = chunk.rfind("\n")
        if newline_pos > max_length * 0.5:
            chunk = chunk[:newline_pos]
        parts.append(chunk.strip())
        text = text[len(chunk):].strip()
    return parts


@dp.message(F.text)
async def handle_question(message: Message) -> None:
    """Обработчик текстовых вопросов: ищем в индексе и отвечаем."""
    if not rag_assistant:
        await message.answer(
            "Бот ещё не готов. Подождите, пока завершится инициализация.")
        return

    query = (message.text or "").strip()
    if not query:
        await message.answer("Пожалуйста, отправь текстовый вопрос.")
        return

    # Сообщаем пользователю, что запрос обрабатывается.
    thinking_message = await message.answer("Думаю…")

    try:
        result = rag_assistant.ask(query)
        answer = result["answer"]
        reasoning = result.get("reasoning", "")

        # Удаляем служебное сообщение перед отправкой ответа.
        await thinking_message.delete()

        # Telegram ограничивает длину одного сообщения 4096 символами.
        for part in split_telegram_text(answer):
            await message.answer(part)
        if reasoning:
            for part in split_telegram_text("Рассуждения: " + reasoning):
                await message.answer(part)
        else:
            await message.answer("Пошаговое рассуждение модели отсутствует.")
    except Exception:
        logger.exception("Ошибка при обработке запроса: %s", query)
        await thinking_message.edit_text(
            "Произошла ошибка при обработке запроса. "
            "Попробуйте переформулировать вопрос или обратитесь позже.")


async def main() -> None:
    """Точка входа: инициализация RAG и запуск polling."""
    global rag_assistant

    logger.info("Инициализация RAG-ассистента...")
    rag_assistant = RAGAssistant()
    logger.info("RAG-ассистент готов.")

    bot = Bot(
        token=TELEGRAM_BOT_TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.MARKDOWN),
    )

    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Бот остановлен по сигналу Ctrl+C.")
        sys.exit(130)
