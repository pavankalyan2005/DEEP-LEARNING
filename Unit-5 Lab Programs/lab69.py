import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,y=load_wine(return_X_y=True); Z=StandardScaler().fit_transform(X); pca=PCA(n_components=2); X2=pca.fit_transform(Z)
print("Original dimensions:",X.shape[1]); print("Dimensions after PCA:",X2.shape[1])
print("Explained variance by PC1 and PC2 (%):",np.round(pca.explained_variance_ratio_*100,2))
print("First five projected observations:"); print(np.round(X2[:5],2))
print("Class labels of first five observations:",y[:5])
