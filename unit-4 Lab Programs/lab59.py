# Lab 59 - Vanishing Gradient Demonstration

gradient = 1.0
print("Vanishing Gradient Across Time Steps")
for step in range(1, 11):
    gradient *= 0.5
    print("Time step {:2d}: gradient = {:.6f}".format(step, gradient))
print("Observation: gradient becomes very small over many time steps.")
