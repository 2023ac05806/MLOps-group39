# Iris MLOps Pipeline

## Features
- MLflow experiment tracking
- FastAPI serving with pydantic validation
- Docker containerization
- GitHub Actions CI/CD
- Logging of predictions

## How to Run
```bash
python src/preprocess.py
python src/train.py
uvicorn app.main:app --reload
```

## Docker
```bash
docker build -t iris-ml-api .
docker run -p 8000:8000 iris-ml-api
