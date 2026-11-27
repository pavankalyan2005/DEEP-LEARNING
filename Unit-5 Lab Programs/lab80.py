import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
X,y=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
model=MLPClassifier(hidden_layer_sizes=(32,16),activation='relu',solver='adam',max_iter=500,random_state=42,early_stopping=True)
model.fit(Xtr,ytr); pred=model.predict(Xte)
print("Application: healthcare classification using a neural network")
print("Test samples:",len(yte)); print("Accuracy: {:.3f}".format(accuracy_score(yte,pred)))
print("Precision: {:.3f}".format(precision_score(yte,pred)))
print("Recall: {:.3f}".format(recall_score(yte,pred)))
print("F1-score: {:.3f}".format(f1_score(yte,pred)))
print("Note: demonstration on a public breast-cancer dataset; not for clinical diagnosis.")
