FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache -r requirements.txt

COPY src/ src/
COPY scripts/ scripts/
COPY config.json .
COPY data/ data/

CMD ["python", "-m", "scripts.train"]
