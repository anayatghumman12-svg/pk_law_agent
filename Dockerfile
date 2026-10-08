FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY settings.py .
COPY agent/ agent/
COPY main.py .
COPY chroma_db/ chroma_db/

EXPOSE 8000

# Hosting platforms inject PORT; fall back to 8000 for local docker run.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]