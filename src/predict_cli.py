import joblib
import numpy as np
import sys
from load_data import load_iris_df

def predict_from_args(model_path='models/logreg.joblib'):
    model = joblib.load(model_path)
    if len(sys.argv) != 5:
        print("Usage: python predict_cli.py sepal_len sepal_wid petal_len petal_wid")
        return
    vals = list(map(float, sys.argv[1:5]))
    pred = model.predict([vals])[0]
    data = load_iris_df()[1]
    print("Predicted class index:", pred)
    print("Predicted species:", data.target_names[pred])

if __name__ == "__main__":
    predict_from_args()
