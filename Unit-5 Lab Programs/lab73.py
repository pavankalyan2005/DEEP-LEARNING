import numpy as np
from sklearn.decomposition import FastICA
rng=np.random.RandomState(12); t=np.linspace(0,10,1500)
voice1=np.sin(2*np.pi*2*t)+0.25*np.sin(2*np.pi*5*t)
voice2=np.sign(np.sin(2*np.pi*1.1*t))*0.7
S=np.c_[voice1,voice2]; S=(S-S.mean(0))/S.std(0)
A=np.array([[1,0.6],[0.5,1.0]]); mixed=S@A.T
ica=FastICA(n_components=2,random_state=12,max_iter=1500); recovered=ica.fit_transform(mixed)
# ICA components are independent up to scale and order; report absolute correlations.
corr=np.corrcoef(np.c_[S,recovered].T)[:2,2:]
print("Microphone mixtures:",mixed.shape); print("Recovered sources:",recovered.shape)
print("Absolute source/recovered correlations:"); print(np.round(np.abs(corr),3))
print("Mean best-match correlation: {:.3f}".format(np.mean(np.max(np.abs(corr),axis=1))))
