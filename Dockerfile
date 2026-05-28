FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY src/        ./src/
COPY configs/    ./configs/
COPY tests/      ./tests/
COPY .env.example .env.example

# mlruns and data are mounted at runtime, not baked into the image
VOLUME ["/app/mlruns", "/app/data"]

ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
