# CPU smoke image for enterprise risk demo (offline path)
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-enterprise.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-enterprise.txt

COPY . .

ENV PYTHONPATH=/app
ENV LLM_PROVIDER=offline

EXPOSE 8501

# Default: Streamlit demo (offline)
CMD ["streamlit", "run", "src/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
