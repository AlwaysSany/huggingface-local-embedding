FROM python:3.13.3-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "huggingface_embedding_server:app", "--host", "0.0.0.0", "--port", "8000"] 