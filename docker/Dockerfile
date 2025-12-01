FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование приложения
COPY app/ ./app/

# Создание пользователя (опционально, для безопасности)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Монтируемая директория для моделей
VOLUME ["/app/models"]

# Порт
EXPOSE 7860

# Команда запуска
CMD ["python", "-m", "app.main"]