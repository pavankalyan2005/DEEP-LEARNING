import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

X = np.array([[1000, 2, 1], [1200, 2, 0], [1500, 3, 1], [1800, 3, 0], [2000, 4, 1], [2200, 4, 0]])
y = np.array([200, 220, 300, 310, 400, 420])

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)

print("Predictions:", predictions)
print("Mean squared error:", mean_squared_error(y, predictions))
print("Mean absolute error:", mean_absolute_error(y, predictions))
print("R-squared:", model.score(X, y))

 