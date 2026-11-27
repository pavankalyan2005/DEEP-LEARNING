from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X,y=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
sc=StandardScaler(); A=sc.fit_transform(Xtr); B=sc.transform(Xte)
base=LogisticRegression(max_iter=3000).fit(A,ytr); acc1=accuracy_score(yte,base.predict(B))
pca=PCA(n_components=.95); C=pca.fit_transform(A); D=pca.transform(B)
reduced=LogisticRegression(max_iter=3000).fit(C,ytr); acc2=accuracy_score(yte,reduced.predict(D))
print("Features before PCA:",X.shape[1]); print("Features after PCA:",pca.n_components_)
print("Accuracy before PCA: {:.2f}%".format(acc1*100)); print("Accuracy after PCA: {:.2f}%".format(acc2*100))
print("Variance retained: {:.2f}%".format(pca.explained_variance_ratio_.sum()*100))
