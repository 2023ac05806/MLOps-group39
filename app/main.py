from fastapi import FastAPI
from app.schema import IrisInput
import joblib
import logging

app = FastAPI()
model = joblib.load("models/model.pkl")

logging.basicConfig(filename='logs/prediction_logs.txt', level=logging.INFO)

@app.post("/predict")
def predict(input_data: IrisInput):
    features = [[
        input_data.sepal_length,
        input_data.sepal_width,
        input_data.petal_length,
        input_data.petal_width,
    ]]
    prediction = model.predict(features)[0]
    logging.info(f"Prediction request: {input_data.dict()}, Prediction: {prediction}")
    return {"prediction": int(prediction)}
