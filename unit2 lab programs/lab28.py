from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
X, y = make_classification(n_samples=1000, n_features=10, n_informative=6, n_redundant=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
for activation in ["relu", "logistic", "tanh"]:
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,8), activation=activation, max_iter=1000, random_state=42))
    model.fit(X_train, y_train)
    print("Activation:", activation)
    print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
    print("Iterations:", model[-1].n_iter_)
