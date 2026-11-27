import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FactorAnalysis
X,y=load_wine(return_X_y=True); Z=StandardScaler().fit_transform(X); k=5
pca=PCA(n_components=k).fit(Z); pca_rec=pca.inverse_transform(pca.transform(Z)); pca_err=np.mean((Z-pca_rec)**2)
fa=FactorAnalysis(n_components=k,random_state=0).fit(Z); fa_scores=fa.transform(Z); fa_rec=fa.mean_+fa_scores@fa.components_; fa_err=np.mean((Z-fa_rec)**2)
print("Original dimensions:",Z.shape[1]); print("Reduced dimensions:",k)
print("PCA reconstruction MSE: {:.4f}".format(pca_err)); print("Factor Analysis reconstruction MSE: {:.4f}".format(fa_err))
print("Lower reconstruction MSE indicates better reconstruction on this dataset.")
