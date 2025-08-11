# Iris MLOps Pipeline

## Features
- Data Preprocessing: src/preprocess.py loads and prepares the Iris dataset.

- Model Training: src/train.py trains multiple models (LogisticRegression and RandomForestClassifier), compares their performance, and saves the best model.

- Experiment Tracking: MLflow is used to log model parameters, metrics, and artifacts for easy comparison.

- API Serving: The model is exposed via a FastAPI service with a /predict endpoint.

- Input Validation: API requests are validated using Pydantic schemas to ensure data integrity.

- Containerization: The entire service is packaged in a Docker container for portability and consistent environments.

- CI/CD: A GitHub Actions workflow lints the code and automatically builds and pushes the Docker image to Docker Hub on every push to the main branch.

- Logging: Prediction requests and outputs are logged to a file with timestamps for basic monitoring.

- Monitoring Integration: The API exposes a /metrics endpoint for collecting performance data with - Prometheus.

- The project architecture is set up for dashboard creation with Grafana and automated model retraining.

## Preprocess and train the model
python src/preprocess.py
python src/train.py

# Run the API with logging
uvicorn app.main:app --reload

# Build the Docker image
docker build -t iris-ml-api .

# Run the container, mapping port 8000
docker run -p 8000:8000 iris-ml-api