from sklearn.datasets import load_iris
import pandas as pd

def load_iris_df():
    data = load_iris(as_frame=True)
    df = data.frame
    df['target_name'] = df['target'].map(dict(enumerate(data.target_names)))
    return df, data

if __name__ == "__main__":
    df, data = load_iris_df()
    print(df.head())
