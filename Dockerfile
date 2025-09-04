# syntax=docker/dockerfile:1

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY artifacts/ artifacts/
COPY src/ src/

EXPOSE 8000

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501"]

