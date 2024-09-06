import numpy as np
class gradient_descent(object) :

    def __init__(self, n_iterations, learning_rate) :
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def initializing_weights(self, n_features) :
        # initialize weights as zeros 
        self.w = np.zeros(n_features)

    def fit(self, X, y) :
        # insert ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        self.initializing_weights(X.shape[1])

        for i in range(self.n_iterations) :
            y_pred = X.dot(self.w)
            mse = 0.5 * (y - y_pred)**2

            grad_w = -(y - y_pred).dot(X)

            self.w -= self.learning_rate*grad_w

    def predict(self, X, y) :
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
    
