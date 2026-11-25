import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
X = np.arange(1, 31).reshape(-1, 1)
y = 4 * X.ravel() + 10 * np.sin(X.ravel())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
for degree in [1, 2, 5, 10]:
    poly = PolynomialFeatures(degree=degree)
    train_x = poly.fit_transform(X_train)
    test_x = poly.transform(X_test)
    model = LinearRegression().fit(train_x, y_train)
    print("Degree:", degree)
    print("Training MSE:", mean_squared_error(y_train, model.predict(train_x)))
    print("Testing MSE:", mean_squared_error(y_test, model.predict(test_x)))
