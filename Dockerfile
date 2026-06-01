FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY src/     ./src/
COPY models/  ./models/
COPY params.yaml .

CMD sh -c "streamlit run src/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"
