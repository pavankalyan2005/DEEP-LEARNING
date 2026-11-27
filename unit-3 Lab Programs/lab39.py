# lab39.py
# Apply convolution and pooling to a sample grayscale surveillance image.
import numpy as np
from scipy.signal import convolve2d

image = np.array([
    [0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0],
    [0, 1, 2, 2, 1, 0],
    [0, 1, 2, 2, 1, 0],
    [0, 0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0]
], dtype=float)
kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
feature_map = convolve2d(image, kernel, mode="valid")
pooled = np.array([
    [np.max(feature_map[i:i+2, j:j+2]) for j in range(0, feature_map.shape[1]-1, 2)]
    for i in range(0, feature_map.shape[0]-1, 2)
])
print("Input image:\n", image)
print("Convolution feature map:\n", feature_map)
print("Max-pooled feature map:\n", pooled)
