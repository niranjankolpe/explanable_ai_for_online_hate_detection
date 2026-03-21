FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements-docker.txt && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]