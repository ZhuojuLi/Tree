import numpy as np
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)

def gradient(y_true, y_pred):
    return y_pred - y_true

def hessian(y_true, y_pred):
    return np.ones_like(y_true)

from sklearn.tree import DecisionTreeRegressor

class SimpleXGBoost:
    def __init__(self, n_estimators=10, max_depth=3, learning_rate=0.1, reg_lambda=1):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.trees = []

    def fit(self, X, y):
        y_pred = np.zeros_like(y, dtype=float)

        for i in range(self.n_estimators):
            g = gradient(y, y_pred)
            h = hessian(y, y_pred)

            # 构造一棵树来拟合负梯度
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, -g / (h + self.reg_lambda))  # 近似二阶 Newton 步长

            # 更新预测
            update = tree.predict(X)
            y_pred += self.learning_rate * update

            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

model = SimpleXGBoost(n_estimators=10, learning_rate=0.1)
model.fit(X, y)
y_pred = model.predict(X)

# 可视化
plt.scatter(X, y, label="True")
plt.scatter(X, y_pred, color='r', label="Predicted")
plt.legend()
plt.title("Simple XGBoost Regression")
plt.show()
