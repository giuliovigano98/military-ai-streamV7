FROM python:3.11-slim

WORKDIR /app

# Dipendenze di sistema per OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Dipendenze Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia codice app
COPY . .

# === Scarica modelli (URL forniti come build-args da GitHub Actions) ===
ARG MOBILENET_URL=""
ARG TANK_URL=""
ARG IFV_URL=""
ARG MOBILENET_NAME="mobilenet_tank_ifv_model.h5"
ARG TANK_NAME="threat_classifier_modelTANK.h5"
ARG IFV_NAME="threat_classifier_modelIFV.h5"

RUN set -eux; \
    if [ -n "$MOBILENET_URL" ]; then curl -L "$MOBILENET_URL" -o "/app/${MOBILENET_NAME}"; fi; \
    if [ -n "$TANK_URL" ]; then curl -L "$TANK_URL" -o "/app/${TANK_NAME}"; fi; \
    if [ -n "$IFV_URL" ]; then curl -L "$IFV_URL" -o "/app/${IFV_NAME}"; fi

# Config Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]
