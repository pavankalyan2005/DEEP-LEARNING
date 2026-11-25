import numpy as np
q_table = np.zeros((4, 2))
learning_rate = 0.1
discount_factor = 0.9
epsilon = 0.2
for episode in range(1000):
    state = np.random.randint(4)
    for step in range(20):
        action = np.random.randint(2) if np.random.random() < epsilon else np.argmax(q_table[state])
        next_state = np.random.randint(4)
        reward = -state + (2 if action == 1 and state >= 2 else 0)
        q_table[state, action] += learning_rate * (reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])
        state = next_state
print("Learned Q-table:")
print(q_table)
print("Best action for each traffic state:", np.argmax(q_table, axis=1))
