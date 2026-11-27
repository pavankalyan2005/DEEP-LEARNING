# lab40.py
# Compare max pooling and average pooling on the same feature map.
import numpy as np

feature_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [2, 1, 8, 7],
    [3, 4, 6, 5]
], dtype=float)

def pooling(data, mode):
    result = []
    for i in range(0, data.shape[0], 2):
        row = []
        for j in range(0, data.shape[1], 2):
            block = data[i:i+2, j:j+2]
            row.append(np.max(block) if mode == "max" else np.mean(block))
        result.append(row)
    return np.array(result)

print("Original feature map:\n", feature_map)
print("Max pooling:\n", pooling(feature_map, "max"))
print("Average pooling:\n", pooling(feature_map, "average"))
