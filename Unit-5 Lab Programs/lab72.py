import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler
rng=np.random.RandomState(0); t=np.linspace(0,8,1000)
s1=np.sin(2*t); s2=np.sign(np.sin(3*t)); S=np.c_[s1,s2]; S=(S-S.mean(0))/S.std(0)
A=np.array([[1,0.5],[0.4,1.0]]); mixed=S@A.T
ica=FastICA(n_components=2,random_state=0,max_iter=1000,tol=0.001); separated=ica.fit_transform(mixed)
print("Mixed signal shape:",mixed.shape); print("Separated component shape:",separated.shape)
print("Estimated mixing matrix:"); print(np.round(ica.mixing_,3))
print("Correlation between recovered components: {:.4f}".format(np.corrcoef(separated.T)[0,1]))
