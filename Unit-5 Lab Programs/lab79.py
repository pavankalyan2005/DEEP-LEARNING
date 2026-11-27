import numpy as np
from sklearn.mixture import GaussianMixture
X=np.array([[20,15],[22,18],[25,20],[23,16],[60,70],[65,75],[68,72],[62,68],[40,35],[42,38],[38,32],[45,40]],float)
gmm=GaussianMixture(n_components=3,random_state=42,n_init=5).fit(X)
labels=gmm.predict(X); probs=gmm.predict_proba(X)
print("Customer data shape:",X.shape); print("Number of clusters:",gmm.n_components)
print("Predicted cluster labels:",labels)
print("Membership probabilities (first five customers):"); print(np.round(probs[:5],3))
print("Model converged:",gmm.converged_)
