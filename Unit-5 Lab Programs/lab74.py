import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
X,y=load_wine(return_X_y=True); Z=StandardScaler().fit_transform(X)
pca=PCA(n_components=3).fit_transform(Z)
ica=FastICA(n_components=3,random_state=0,max_iter=2000,tol=.001).fit_transform(Z)
print("Original feature count:",X.shape[1]); print("PCA component shape:",pca.shape); print("ICA component shape:",ica.shape)
print("PCA explained variance (%):",np.round(PCA(n_components=3).fit(Z).explained_variance_ratio_*100,2))
print("PCA component correlations (off-diagonal):",np.round(np.corrcoef(pca.T)[0,1],4))
print("ICA component correlations (off-diagonal):",np.round(np.corrcoef(ica.T)[0,1],4))
