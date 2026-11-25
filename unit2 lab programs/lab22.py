from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42)
grid = GridSearchCV(model, {"n_estimators": [50,100], "max_depth": [None,5,10]}, cv=3, scoring="accuracy")
grid.fit(X_train, y_train)
print("Grid search:", grid.best_params_)
print("Grid accuracy:", accuracy_score(y_test, grid.predict(X_test)))
random_search = RandomizedSearchCV(model, {"n_estimators": [50,100,150], "max_depth": [None,5,10], "min_samples_split": [2,5,10]}, n_iter=5, cv=3, scoring="accuracy", random_state=42)
random_search.fit(X_train, y_train)
print("Random search:", random_search.best_params_)
print("Random accuracy:", accuracy_score(y_test, random_search.predict(X_test)))
try:
    from skopt import BayesSearchCV
    bayes = BayesSearchCV(model, {"n_estimators": (50,150), "max_depth": (3,15)}, n_iter=5, cv=3, scoring="accuracy", random_state=42)
    bayes.fit(X_train, y_train)
    print("Bayesian optimization:", bayes.best_params_)
    print("Bayesian accuracy:", accuracy_score(y_test, bayes.predict(X_test)))
except ImportError:
    print("Install scikit-optimize: pip install scikit-optimize")

 