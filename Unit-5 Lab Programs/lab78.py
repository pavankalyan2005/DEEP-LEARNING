import numpy as np
rng=np.random.default_rng(42)
mean=np.array([10.0,20.0,30.0]); cov=np.array([[4,1,.5],[1,9,1],[.5,1,16]])
X=rng.multivariate_normal(mean,cov,size=1000); est_mean=X.mean(axis=0); est_cov=np.cov(X,rowvar=False)
print("Generated data shape:",X.shape)
print("Specified mean:",mean); print("Estimated mean:",np.round(est_mean,2))
print("Specified covariance diagonal:",np.diag(cov))
print("Estimated covariance diagonal:",np.round(np.diag(est_cov),2))
print("Evaluation: estimated parameters should be close to the specified parameters for a large sample.")
