import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


def train_and_log(model, model_name, X_train, y_train, X_test, y_test):
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("accuracy", acc)

        model_path = f"models/{model_name}.pkl"
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(model, model_name)

        print(f"{model_name} accuracy: {acc}")
        return model, acc


# Load dataset
df = pd.read_csv("data/iris.csv")
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

mlflow.set_experiment("iris_classification")

model1, acc1 = train_and_log(
    LogisticRegression(max_iter=200),
    "LogisticRegression", X_train, y_train, X_test, y_test
)
model2, acc2 = train_and_log(
    RandomForestClassifier(), "RandomForest", X_train, y_train, X_test, y_test
)

# Save best model
best_model = model1 if acc1 >= acc2 else model2
joblib.dump(best_model, "models/model.pkl")
