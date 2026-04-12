FROM python:3.10-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY src/ ./src/
COPY models/ ./models/
COPY params.yaml .

CMD sh -c "streamlit run src/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"