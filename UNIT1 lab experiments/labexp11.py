import math

low_risk = 8
high_risk = 4
total = low_risk + high_risk
probabilities = [low_risk / total, high_risk / total]

entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

print("Low-risk probability:", probabilities[0])
print("High-risk probability:", probabilities[1])
print("Entropy:", entropy, "bits")
