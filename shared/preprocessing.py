import numpy as np


class ZScoreScaler:
    """Estandariza features columna a columna: (X - mean) / std.

    Usa std muestral.
    Las columnas con std=0 se reemplazan por 1 para evitar división por cero.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X * self.std_ + self.mean_
