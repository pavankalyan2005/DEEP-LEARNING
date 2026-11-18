import numpy as np
from scipy.stats import normaltest

np.random.seed(42)
temperatures = np.random.normal(loc=25, scale=4, size=1000)

print("Sample mean:", np.mean(temperatures))
print("Sample standard deviation:", np.std(temperatures))
statistic, p_value = normaltest(temperatures)
print("Normality test statistic:", statistic)
print("P-value:", p_value)
if p_value > 0.05:
    print("Sample is consistent with a normal distribution.")
else:
    print("Evidence suggests deviation from a normal distribution.")

 