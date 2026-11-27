import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
X=np.array([[8,9,2,3],[7,8,3,4],[9,8,1,2],[2,3,9,8],[3,2,8,9],[4,3,7,8],[8,7,2,3],[2,4,8,7]],float)
Z=StandardScaler().fit_transform(X); fa=FactorAnalysis(n_components=2,random_state=1).fit(Z)
print("Customer dataset shape:",X.shape)
print("Attributes: purchase frequency, spend, discount use, browsing")
print("Factor loadings:"); print(np.round(fa.components_.T,3))
print("Interpretation: attributes with larger absolute loadings contribute more to each latent factor.")
