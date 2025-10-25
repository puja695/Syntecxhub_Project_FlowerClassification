import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from load_data import load_iris_df
import train

def plot_confusion(saved_model_path, X_test, y_test, title):
    model = joblib.load(saved_model_path)
    preds = model.predict(X_test)
    cm = confusion_matrix(y_test, preds)
    classes = load_iris_df()[1].target_names
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(title)
    plt.show()
    print(classification_report(y_test, preds, target_names=classes))

if __name__ == "__main__":
    X_test, y_test, results, data = train.train_and_save()
    plot_confusion('models/logreg.joblib', X_test, y_test, 'Logistic Regression Confusion Matrix')
    plot_confusion('models/dtree.joblib', X_test, y_test, 'Decision Tree Confusion Matrix')
