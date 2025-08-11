import joblib


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    # Load trained model
    model = joblib.load("model.pkl")

    # Prepare input
    input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Make prediction
    prediction = model.predict(input_data)

    return prediction[0]


if __name__ == "__main__":
    # Sample input
    result = predict_iris(5.1, 3.5, 1.4, 0.2)
    print(f"Predicted class: {result}")