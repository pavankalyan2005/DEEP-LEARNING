import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
X=np.array([[80,85,75,90],[70,72,68,75],[90,92,88,95],[60,65,58,62],[85,88,82,91],[65,68,62,70],[95,96,90,98],[75,78,72,80]],float)
Z=StandardScaler().fit_transform(X); fa=FactorAnalysis(n_components=2,random_state=0).fit(Z); scores=fa.transform(Z)
print("Student records:",X.shape[0]); print("Observed attributes: marks, attendance, assignments, participation")
print("Factor loading matrix:"); print(np.round(fa.components_.T,3))
print("Latent factor scores (first five students):"); print(np.round(scores[:5],3))
