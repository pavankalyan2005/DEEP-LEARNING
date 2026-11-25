import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
X = np.array([[1000,2],[1200,2],[1500,3],[1800,3],[2000,4],[2200,4],[2500,4],[2800,5],[3000,5],[3200,5]], dtype=float)
y = np.array([200,220,300,330,390,420,470,510,550,580], dtype=float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(16,8), max_iter=5000, random_state=42))
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print("Actual prices:", y_test)
print("Predicted prices:", predictions)
print("Mean squared error:", mean_squared_error(y_test, predictions))
print("Mean absolute error:", mean_absolute_error(y_test, predictions))
