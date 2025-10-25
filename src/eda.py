import seaborn as sns
import matplotlib.pyplot as plt
from load_data import load_iris_df

def run_eda():
    df, data = load_iris_df()
    print(df.describe())
    sns.pairplot(df, vars=data.feature_names, hue='target_name', corner=True)
    plt.suptitle('Pairplot of Iris features', y=1.02)
    plt.show()

if __name__ == "__main__":
    run_eda()
