import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X=np.array([[45,120,80,5.2],[50,130,85,5.8],[35,110,75,4.9],[60,150,95,7.1],[55,140,90,6.5],[40,115,78,5.0]],float)
Z=StandardScaler().fit_transform(X); pca=PCA(n_components=2); reduced=pca.fit_transform(Z)
print("Original dataset shape:",X.shape)
print("Reduced dataset shape:",reduced.shape)
print("Explained variance ratio:",np.round(pca.explained_variance_ratio_,4))
print("Transformed features:")
print(np.round(reduced,3))
