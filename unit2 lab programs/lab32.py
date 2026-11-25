from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import log_loss, accuracy_score, classification_report
X, y = make_classification(n_samples=1000, n_features=8, n_informative=5, weights=[0.9,0.1], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,8), max_iter=1500, random_state=42))
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)
print("Binary cross-entropy loss:", log_loss(y_test, probabilities))
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))
