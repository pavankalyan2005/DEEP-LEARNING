import numpy as np

P = np.array([0.40, 0.35, 0.25])
Q = np.array([0.50, 0.30, 0.20])

kl_divergence = np.sum(P * np.log(P / Q))
print("KL divergence (nats):", kl_divergence)
