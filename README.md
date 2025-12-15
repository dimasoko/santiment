# VK Sentiment Analysis

Анализ тональности постов из социальной сети ВКонтакте с использованием RuBERT и визуализацией результатов.

## Описание проекта

Проект выполняет автоматический анализ тональности русскоязычных постов из VK с помощью предобученной модели RuBERT. Включает очистку текста от стоп-слов, классификацию на POSITIVE/NEGATIVE/NEUTRAL и генерацию визуализаций.

### Возможности

- Автоматическая классификация тональности текстов (позитивная, негативная, нейтральная)
- Очистка текста от ссылок, хэштегов, стоп-слов и специальных символов
- Статистический анализ распределения сантиментов
- Генерация облаков слов (word clouds) по категориям
- Анализ длины текстов в зависимости от тональности
- Топ наиболее частых слов для каждой категории

## Технологии

- Python 3.8+
- Transformers (Hugging Face) — RuBERT модель для анализа тональности
- Pandas — обработка данных
- Matplotlib — визуализация
- WordCloud — генерация облаков слов
- PyTorch — backend для трансформеров

## Требования

- Python 3.8 или выше
- Windows/Linux/macOS
- 2+ GB RAM для загрузки модели
- Интернет для первой загрузки RuBERT (~500 MB)

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd santiment
```

### 2. Создание виртуального окружения

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Подготовка данных

Скачайте датасет [VK Filtered Social Posts with Engagement Metadata](https://www.kaggle.com/datasets/nooruzbekt/vk-social-media-posts-metadata) с Kaggle и поместите файл `vk_filtered_posts_metadata.csv` в корень проекта.

### 5. Запуск анализа

```bash
python sentiment_vk.py
```

При первом запуске автоматически загрузится модель RuBERT (~500 MB).

## Структура проекта

```
santiment/
├── .venv/                          # Виртуальное окружение (не коммитится)
├── sentiment_vk.py                 # Основной скрипт анализа
├── vk_filtered_posts_metadata.csv  # Исходный датасет (скачать с Kaggle)
├── vk_with_sentiment.csv           # Результаты с метками тональности
├── requirements.txt                # Зависимости Python
├── .gitignore                      # Исключения для Git
├── .gitattributes                  # Настройки окончаний строк
└── README.md                       # Документация

# Генерируемые файлы:
├── sentiment_distribution.png      # Графики распределения сантиментов
├── text_length_by_sentiment.png    # Анализ длины текстов
├── wc_all.png                      # Облако слов (все посты)
├── wc_positive.png                 # Облако слов (позитивные)
├── wc_negative.png                 # Облако слов (негативные)
└── wc_neutral.png                  # Облако слов (нейтральные)
```

## Результаты

После выполнения скрипта будут созданы:

### Файлы данных
- `vk_with_sentiment.csv` — исходный датасет + колонки `sent_label` (метка) и `sent_score` (уверенность модели)

### Визуализации
- **sentiment_distribution.png** — bar chart и pie chart распределения категорий
- **text_length_by_sentiment.png** — boxplot длины текстов по сантиментам
- **wc_all.png** — общее облако слов
- **wc_positive.png**, **wc_negative.png**, **wc_neutral.png** — облака по категориям

### Консольный вывод
- Статистика по количеству и процентам каждой категории
- Средняя уверенность модели
- Топ-15 слов для каждой тональности
- Примеры обработанных постов

## Настройка

### Изменение параметров в sentiment_vk.py

```python
CSV_PATH = "vk_filtered_posts_metadata.csv"  # Путь к датасету
batch_size = 32                               # Размер батча для модели
top_n = 15                                    # Количество топ-слов
```

### Добавление стоп-слов

Отредактируйте список `RUSSIAN_STOPWORDS` в начале файла для более точной фильтрации.

## Пример использования результатов

```python
import pandas as pd

# Загрузка результатов
df = pd.read_csv("vk_with_sentiment.csv")

# Фильтрация только позитивных постов
positive = df[df["sent_label"] == "POSITIVE"]

# Сортировка по уверенности модели
top_positive = df[df["sent_label"] == "POSITIVE"].nlargest(10, "sent_score")
print(top_positive[["Text", "sent_score"]])
```

## Решение проблем

### Ошибка при установке fasttext
Используйте `fasttext-wheel`:
```bash
pip install fasttext-wheel
```

### Ошибка "Microsoft Visual C++ required"
Установите [Build Tools for Visual Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

### Модель не скачивается
Проверьте интернет-соединение или используйте VPN (если Hugging Face заблокирован)

## Лицензия

MIT

## Автор

Проект создан в рамках курса по анализу данных
