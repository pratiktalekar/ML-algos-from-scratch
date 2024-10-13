import numpy as np

class logistic_regression() :

    def __init__(self, learning_rate, iterations) :
        self.lr = learning_rate
        self.iterations = iterations

    def sigmoid(self, x) :
        return 1/(1+ np.exp(-x))

    def fit(self, X, y) :
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = bias

        for i in range(self.iterations) :
            
            linear_data = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(linear_data)

            dw = (1/n_samples) * np.dot(X.T, (y-y_pred))
            db = (1/n_samples) * np.dot(x.T, (y - y_pred))

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict(self, X) :
        linear_data = np.dot(X.T, self.weights) + self.bias
        y = self.sigmoid(linear_data)
        y_class = [1 if i>0.5 else 0 for i in linear_data]
        return np.array(y_class)
    

        
            
        