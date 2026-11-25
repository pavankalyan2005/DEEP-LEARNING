import numpy as np
value32 = np.float32(0.1)
value64 = np.float64(0.1)
print("Float32 value:", value32)
print("Float64 value:", value64)
print("Float32 addition:", value32 + np.float32(0.2))
print("Float64 addition:", value64 + np.float64(0.2))
print("Float32 large plus small:", np.float32(1e8) + np.float32(1))
print("Float64 large plus small:", np.float64(1e8) + np.float64(1))

 