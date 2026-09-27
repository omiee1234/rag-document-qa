FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
RUN pip install --no-cache-dir -e .

# Build the default index at image build time so the container is ready to serve.
RUN ragqa ingest --docs data/docs --index index.pkl --embedder tfidf

ENV RAGQA_INDEX_PATH=/app/index.pkl
EXPOSE 8000

CMD ["uvicorn", "ragqa.api:app", "--host", "0.0.0.0", "--port", "8000"]
