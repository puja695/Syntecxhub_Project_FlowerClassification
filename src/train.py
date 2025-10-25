import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from load_data import load_iris_df
import os

def train_and_save():
    df, data = load_iris_df()
    X = df[data.feature_names]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        'logreg': LogisticRegression(max_iter=200),
        'dtree': DecisionTreeClassifier(random_state=42)
    }

    results = {}
    os.makedirs('models', exist_ok=True)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        results[name] = {'model': model, 'accuracy': acc, 'preds': preds}
        joblib.dump(model, f'models/{name}.joblib')
        print(f"{name} accuracy: {acc:.4f}")
    return X_test, y_test, results, data

if __name__ == "__main__":
    train_and_save()
