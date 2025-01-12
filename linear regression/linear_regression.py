import numpy as np

class l2_regularization() :
    def __init__(self, alpha):
        self.alpha = alpha
    
    
        
class regression(object) :

    def __init__(self, n_iterations, learning_rate) :
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initializing_weights(self, n_features) :
        # initialize weights as zeros 
        self.w = np.zeros(n_features)

    def fit(self, X, y) :
        # insert ones for bias weights
        X = np.insert(X, 0, np.ones(X.shape[0], 1), axis=1)
        self.initializing_weights(X.shape[1])

        for i in range(self.n_iterations):
            y_pred = X.dot(self.w)
            mse = 0.5 * (y - y_pred)**2

            grad_w = -(y - y_pred).dot(X)

            self.w -= self.learning_rate*grad_w

    def predict(self, X, y) :
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
    
class normal_equation(regression):
    def __init__(self, n_iterations, learning_rate, gradient_descent=True):
        super().__init__(n_iterations, learning_rate)
        self.gradient_descent = gradient_descent
        self.w = None

    def fit(self, X, y):
        if not self.gradient_descent:
            X_new = np.c_(np.ones(X.shape[0], 1), X)
            self.initializing_weights(X_new.shape[1])
            self.w = np.linalg.inv(X_new.T.dot(X_new)).dot(X_new.T).dot(y)
        else:
            super().fit(self, X, y)

    def predict(self, X):
        X_new = np.c_(np.ones(1, 1), X)
        y_pred = self.w * X_new
        return y_pred
'''
1. np.insert(arr, obj, values, axis) 
    arr is in which we have to insert the values, obj is index before which we have to insert values
    values is what we have to insert in arr , it can be any integer, sequence of integers.
    axis is used in case of 2-d array , it is used to insert values columnwise(axis=1) or rowwise(axis=0)
'''