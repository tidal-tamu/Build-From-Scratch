import numpy as np

class LinearRegression:

    def __init__(self):
        self.coef_ = []

    def __format_x(self, X):
        """Add a column of 1s to account for y-intercepts"""
        X = X.values 
        X = np.column_stack((np.ones(X.shape[0]),X))
        return X
    
    def fit(self, X, y):
        X = self.__format_x(X)
        y = np.array(y)
        transpose_x = np.transpose(X)
        self.coef_ = np.matmul(
            np.linalg.inv(
                np.matmul(
                    transpose_x,
                    X
                )
            ),
            np.matmul(
                transpose_x,
                y
            )
        )

    def predict(self, X):
        return np.matmul(
            self.__format_x(X),
            self.coef_
        )

    def score(self, X, y):
        predictions = self.predict(X)
        y = y.tolist()
        RSS = sum((actual-predicted)**2 for actual, predicted in zip(y, predictions))
        mean_y = sum(y)/len(y)
        TSS = sum((actual-mean_y)**2 for actual in y)
        return 1 - RSS/TSS