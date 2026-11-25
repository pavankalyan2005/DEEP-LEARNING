import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X = np.array([[1,2,3],[2,1,2],[3,4,1],[4,3,2],[5,6,3],[6,5,4],[7,8,5],[8,7,6]], dtype=float)
y = np.array([100,120,150,170,210,230,270,290], dtype=float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
models = {"Linear": LinearRegression(), "L1": Lasso(alpha=0.1), "L2": Ridge(alpha=0.1)}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(name, "Coefficients:", model.coef_)
    print(name, "MSE:", mean_squared_error(y_test, model.predict(X_test)))
