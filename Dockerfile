
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Add gdown for robust Google Drive downloads ---
RUN pip install --no-cache-dir gdown

# App code
COPY . .

# ====== Model download (works with Google Drive and normal URLs) ======
# Build args provided by GitHub Actions
ARG MOBILENET_URL=""
ARG TANK_URL=""
ARG IFV_URL=""
ARG MOBILENET_NAME="mobilenet_tank_ifv_model.h5"
ARG TANK_NAME="threat_classifier_modelTANK.h5"
ARG IFV_NAME="threat_classifier_modelIFV.h5"

# Make them available to the Python snippet
ENV MOBILENET_URL=${MOBILENET_URL} \
    TANK_URL=${TANK_URL} \
    IFV_URL=${IFV_URL} \
    MOBILENET_NAME=${MOBILENET_NAME} \
    TANK_NAME=${TANK_NAME} \
    IFV_NAME=${IFV_NAME}

# Download with gdown if URL is Google Drive; otherwise curl -L
RUN python - <<'PY'
import os, subprocess
def fetch(url, out):
    if not url: 
        print(f"skip {out} (empty URL)"); return
    if "drive.google.com" in url:
        import gdown
        print(f"[gdown] {url} -> {out}")
        gdown.download(url, out, quiet=False, fuzzy=True)
    else:
        print(f"[curl ] {url} -> {out}")
        subprocess.run(["bash","-lc", f'curl -fsSL "{url}" -o "{out}"'], check=True)

pairs = [
    (os.getenv("MOBILENET_URL"), os.getenv("MOBILENET_NAME")),
    (os.getenv("TANK_URL"),      os.getenv("TANK_NAME")),
    (os.getenv("IFV_URL"),       os.getenv("IFV_NAME")),
]
for url, name in pairs:
    if name:
        fetch(url, f"/app/{name}")
print("Downloaded files:")
subprocess.run(["bash","-lc","ls -lh /app | grep -E '\\.h5$' || true"])
PY

# Streamlit config
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501
CMD ["streamlit","run","app.py","--server.port=8501","--server.address=0.0.0.0"]

