from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
X,y=load_wine(return_X_y=True); Z=StandardScaler().fit_transform(X); pca=PCA().fit(Z)
cumulative=pca.explained_variance_ratio_.cumsum(); k=int((cumulative>=.95).argmax()+1)
print("Total original features:",X.shape[1]); print("Components required for 95% variance:",k)
print("Cumulative explained variance: {:.2f}%".format(cumulative[k-1]*100))
print("Individual explained variance (%):",(pca.explained_variance_ratio_[:k]*100).round(2))
