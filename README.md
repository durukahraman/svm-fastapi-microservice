# SVM FastAPI Microservice

Basit bir FastAPI servisi, SVM modeli ile tahmin yapar.

## Çalıştırma

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

