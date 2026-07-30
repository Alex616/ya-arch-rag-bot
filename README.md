# Векторный индекс для `knowledge_base`

Проект создаёт локальный векторный индекс из обезличенных статей в
`knowledge_base/` с помощью **ChromaDB** и локальной CPU-модели эмбеддингов.

Имена файлов в `knowledge_base/` переименованы в нумерованные
`article_NNN.txt`, чтобы не содержать идентифицирующих имён персонажей и
названий. Старое соответствие имён не сохраняется.

## Что сделано

- Все статьи из `knowledge_base/` разбиты на семантические чанки через
  `RecursiveCharacterTextSplitter` из LangChain.
- Эмбеддинги сгенерированы локально на CPU с помощью модели Hugging Face `sentence-transformers`.
- Векторы и метаданные сохранены в персистентную коллекцию ChromaDB (`chroma.db/`).
- Добавлен `rag.py` — reusable RAG-ассистент:
  embeddings → Chroma → RetrievalQA → YandexGPT (OpenAI-совместимый API),
  готовый к импорту в Telegram-бот.
- Добавлен `query.py` — CLI-скрипт, демонстрирующий работу `rag.py`.

## Файлы

| Файл / директория | Назначение |
| ----------------- | ---------- |
| `knowledge_base/` | Исходные статьи (31 очищенный `.txt`-файл). |
| `build_index.py` | Собирает чанки и сохраняет индекс в `chroma.db/`. |
| `chroma.db/` | Персистентная БД ChromaDB (генерируется). |
| `query.py` | CLI-скрипт: RAG-запрос + ответ от YandexGPT. |
| `rag.py` | Reusable RAG-модуль, который можно импортировать в Telegram-бот. |
| `few_shot_examples.py` | Статические few-shot примеры для промпта RetrievalQA. |
| `.env.example` | Пример переменных окружения для работы с YandexGPT. |
| `pyproject.toml` | Зависимости Poetry. |

## Используемая модель

- **Модель эмбеддингов:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- **Почему эта модель:** компактная мультиязычная sentence-transformer,
  которая хорошо работает на CPU и подходит для русского текста. Это
  соответствует требованию запуска на локальной машине из
  `Project_template.md`.
- **Векторная БД:** ChromaDB
- **Размер чанка:** 512 символов, перекрытие 128 символов
- **Фильтрация:** чанки короче 50 символов отбрасываются (убирает артефакты разметки).

## Статистика индекса

- **База знаний:** 31 обезличенный файл в `knowledge_base/`
  (`article_001.txt` … `article_031.txt`)
- **Всего символов:** ~480 398
- **Чанков в индексе:** 1369
- **Оборудование:** локальный CPU (Apple Silicon, модель запущена с `device="cpu"`)

## Время генерации (локальный CPU)

Замер на рабочей машине:

```text
Инициализация модели : 3.75 с   (после первого скачивания)
Чанкинг              : 0.03 с
Построение индекса   : 13.04 с
Общее время          : 16.83 с
```

> При первом запуске модель скачивается с Hugging Face (≈ 440 МБ). Повторные
> запуски используют кэш и занимают ≈ 15–20 секунд.

## Запуск

### 1. Установка зависимостей

```bash
poetry install
```

### 2. Построение индекса

```bash
poetry run python build_index.py
```

Скрипт создаст/перезапишет директорию `chroma.db/`.

### 3. Запрос через LLM (YandexGPT)

Скрипт `query.py` использует `rag.RAGAssistant`, который:

1. Превращает запрос в эмбеддинг той же моделью, что и `build_index.py`.
2. Находит ближайшие чанки в ChromaDB.
3. Формирует промпт с фрагментами и вопросом.
4. Отправляет промпт в YandexGPT через OpenAI-совместимый endpoint.
5. Возвращает готовый ответ и список использованных источников.

#### 3.1. Настройка окружения

```bash
cp .env.example .env
# Отредактируйте .env и вставьте ваш YANDEX_API_KEY
```

Обязательная переменная:

- `YANDEX_API_KEY` — API-ключ Yandex Cloud.

Опциональные переменные:

- `YANDEX_BASE_URL` — endpoint (`https://llm.api.cloud.yandex.net/foundationModels/v1`).
- `YANDEX_MODEL` — модель (`yandexgpt-lite`, `yandexgpt`, …).
- `YANDEX_TEMPERATURE`, `YANDEX_MAX_TOKENS`.

#### 3.2. Запуск

Пример с запросом по умолчанию:

```bash
poetry run python query.py
```

Свой запрос:

```bash
poetry run python query.py "Кто такой Мясник из Беловежа?"
```

Количество чанков в контексте (`k`):

```bash
poetry run python query.py --k 7 "Кто такой Мясник из Беловежа?"
```

#### 3.3. Few-shot + Chain-of-Thought prompting

Промпт для `RetrievalQA` строится так, чтобы модель сначала проводила
пошаговое рассуждение (CoT), а затем выдавала финальный ответ после
маркера `Ответ:`.

Файл `few_shot_examples.py` содержит статические примеры:

- как рассуждать на основе одного релевантного фрагмента;
- как синтезировать рассуждение из нескольких фрагментов;
- как корректно отказываться отвечать, если в фрагментах нет нужной
  информации.

В консольном скрипте `query.py` можно посмотреть само рассуждение:

```bash
poetry run python query.py --reasoning "Кто такой Мясник из Беловежа?"
```

В Telegram-боте финальный ответ доступен через `result["answer"]`.

Чтобы добавить новые примеры или убрать few-shot, отредактируйте
`few_shot_examples.py`. В коде `RAGAssistant` можно передать свой набор
примеров или отключить их:

```python
# со своими примерами
assistant = RAGAssistant(few_shot_examples=custom_examples)

# без few-shot
assistant = RAGAssistant(few_shot_examples=None)
```

## Использование в Telegram-боте

`RAGAssistant` можно создать один раз при старте бота и вызывать из хендлера:

```python
from aiogram import Bot, Dispatcher, types
from rag import RAGAssistant

bot = Bot(token="YOUR_BOT_TOKEN")
dp = Dispatcher()

# Инициализируем один раз — модель эмбеддингов и Chroma загружаются здесь.
assistant = RAGAssistant()


@dp.message()
async def handle(message: types.Message) -> None:
    if not message.text:
        return
    result = assistant.ask(message.text)
    # Можно добавить сноски с источниками из result["sources"]
    await message.answer(result["answer"])


if __name__ == "__main__":
    dp.run_polling(bot)
```

> Важно: не создавайте `RAGAssistant` внутри каждого хендлера сообщения —
> иначе эмбеддинг-модель будет перезагружаться на каждый запрос.

## Пример запроса и ответ

**Запрос:** `Кто такой Мясник из Беловежа и чем он занимается?`

**Использованные чанки (top-5):**

```text
1. article_010.txt: Дмитрий из Рымник, прозванный Мясником из Беловежа ...
2. article_014.txt: Алексей Мельников ... профессиональный наёмный убийца ...
3. article_015.txt: Сорокин, который Следопыт волен отдать барду ...
...
```

**Ответ LLM** (зависит от модели):

```text
Мясник из Беловежа — Дмитрий из Рымник, профессиональный охотник на чудовищ...
```

## Примечания

- `chroma.db/` и локальные кэши моделей исключены из Git (см. `.gitignore`).
- Для продакшена или более высокого качества поиска по-русски можно заменить
  модель на `intfloat/multilingual-e5-large` (тяжелее и медленнее на CPU) или
  использовать Yandex Embeddings API, как описано в `Project_template.md`.
