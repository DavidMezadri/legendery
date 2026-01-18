FROM python:3.12-slim

WORKDIR /app

# Dependências do sistema (ffmpeg e tesseract são essenciais)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-por \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]