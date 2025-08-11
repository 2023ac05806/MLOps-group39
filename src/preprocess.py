from sklearn.datasets import load_iris


def load_and_preprocess():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv('data/iris.csv', index=False)
    return df
