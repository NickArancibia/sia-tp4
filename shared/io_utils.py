import pandas as pd
import numpy as np


def load_europe(path):
    """Lee europe.csv y devuelve (countries, X, feature_names).

    - countries: lista de 28 strings con los nombres de los países.
    - X: ndarray (28, 7) con las 7 variables numéricas, sin la columna Country.
    - feature_names: lista de 7 strings con los nombres de las variables.
    """
    df = pd.read_csv(path)
    countries = df["Country"].tolist()
    feature_names = [c for c in df.columns if c != "Country"]
    X = df[feature_names].to_numpy(dtype=np.float64)
    return countries, X, feature_names
