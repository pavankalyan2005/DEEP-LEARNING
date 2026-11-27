# Lab 50 - Character Level RNN Text Generation
# Learns character patterns from a small sample text.

import numpy as np

text = "deep learning uses recurrent neural networks. "
chars = sorted(set(text))
to_id = {c: i for i, c in enumerate(chars)}
to_char = {i: c for c, i in to_id.items()}
v = len(chars)
rng = np.random.default_rng(2)
Wx = rng.normal(0, 0.15, (v, 16))
Wh = rng.normal(0, 0.15, (16, 16))
Why = rng.normal(0, 0.15, (16, v))
h = np.zeros(16)
for ch in text:
    x = np.zeros(v); x[to_id[ch]] = 1
    h = np.tanh(x @ Wx + h @ Wh)

# Generate characters by repeatedly applying the learned recurrent state.
current = "d"
generated = current
for _ in range(40):
    x = np.zeros(v); x[to_id[current]] = 1
    h = np.tanh(x @ Wx + h @ Wh)
    probs = np.exp(h @ Why - np.max(h @ Why))
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    current = to_char[idx]
    generated += current

print("Character Level RNN")
print("Training text:", text.strip())
print("Generated text:", generated)
