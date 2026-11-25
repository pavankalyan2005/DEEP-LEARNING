learning_rates = [0.01,  0.1, 0.5, 1.1]
for rate in learning_rates:
    parameter = 10.0
    for epoch in range(100):
        parameter -= rate * 2 * (parameter - 2)
    print("Learning rate:", rate)
    print("Final parameter:", parameter)
    print("Final loss:", (parameter - 2) ** 2)
