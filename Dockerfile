FROM python:3.11-slim

ARG IMAGE_TAG=v1

LABEL org.opencontainers.image.title="California Housing ML"
LABEL org.opencontainers.image.version="${IMAGE_TAG}"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY config.json .
COPY data/ data/

CMD ["python", "-m", "scripts.train"]