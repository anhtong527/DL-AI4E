import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LinearRegression():
    def init(self):
        pass
    
    def fit(self, X, y):
        self._X_init = X
        self._X = np.hstack((np.ones((X.shape[0], 1)), X))
        self._y = np.array(y).reshape(-1, 1)
        
        self._W = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(self._X), self._X)), np.transpose(self._X)), self._y)        
    
    def coef(self):
        # Return the coeficients
        return self._W
    
    def predict(self, x_new):
        x_new = np.hstack((np.ones((x_new.shape[0], 1)), x_new))
        return np.matmul(x_new, self._W)
    
    def plot(self):
        if self._X_init.shape[1] > 1:
            raise Exception('Only for simple LR!')
        
        self._X_init_plot = np.asarray(self._X_init).reshape(-1)
        y_pred = self.predict(np.array(self._X_init))
        
        plt.scatter(self._X_init_plot, self._y)
        plt.plot(self._X_init_plot, y_pred, color='r')


data = pd.read_csv('data_linear.csv').to_numpy()
x = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)
model.plot()