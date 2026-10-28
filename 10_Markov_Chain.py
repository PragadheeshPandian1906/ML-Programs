"""
Experiment 10: Markov Chain
Dataset: Custom state-transition sequence (created in this program)
Purpose: Estimate transition probabilities and predict future states.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Define observed sequence of weather states.
# S = Sunny, C = Cloudy, R = Rainy
states = ["Sunny", "Cloudy", "Rainy"]
observed_sequence = [
    "Sunny", "Sunny", "Cloudy", "Rainy", "Cloudy",
    "Sunny", "Cloudy", "Cloudy", "Rainy", "Rainy",
    "Cloudy", "Sunny", "Sunny", "Cloudy", "Rainy",
    "Cloudy", "Sunny", "Cloudy", "Rainy", "Cloudy",
    "Sunny", "Sunny", "Cloudy", "Rainy", "Cloudy",
    "Cloudy", "Sunny", "Rainy", "Cloudy", "Sunny"
]

print("Experiment: Markov Chain")
print("Dataset source: Custom weather-state sequence")
print("Number of observations:", len(observed_sequence))
print("States:", states)

# 2. Count transitions between consecutive observations
transition_counts = pd.DataFrame(
    0, index=states, columns=states, dtype=int
)

for current_state, next_state in zip(
    observed_sequence[:-1], observed_sequence[1:]
):
    transition_counts.loc[current_state, next_state] += 1

print("\nTransition Count Matrix:")
print(transition_counts)

# 3. Convert transition counts to transition probabilities.
# Each row is normalized to sum to 1.
row_totals = transition_counts.sum(axis=1).replace(0, np.nan)
transition_matrix = transition_counts.div(row_totals, axis=0).fillna(0)

print("\nTransition Probability Matrix:")
print(transition_matrix.round(3))
print("\nRow sums (should be 1 for states with observed outgoing transitions):")
print(transition_matrix.sum(axis=1).round(3))

# 4. Predict the next state distribution from a starting state
starting_state = "Sunny"
steps = 1
initial_distribution = np.zeros(len(states))
initial_distribution[states.index(starting_state)] = 1.0

distribution = initial_distribution.copy()
for _ in range(steps):
    distribution = distribution @ transition_matrix.to_numpy()

print(f"\nStarting state: {starting_state}")
print(f"Predicted state probabilities after {steps} step(s):")
for state, probability in zip(states, distribution):
    print(f"{state}: {probability:.3f}")

most_likely_state = states[int(np.argmax(distribution))]
print("Most likely next state:", most_likely_state)

# 5. Predict distributions for multiple steps
print("\nState probabilities from Sunny over the next 5 steps:")
multi_step = initial_distribution.copy()
multi_step_results = []
for step in range(1, 6):
    multi_step = multi_step @ transition_matrix.to_numpy()
    multi_step_results.append(multi_step.copy())
    print(
        f"Step {step}: "
        + ", ".join(
            f"{state}={prob:.3f}"
            for state, prob in zip(states, multi_step)
        )
    )

# 6. Visualize transition probability matrix
plt.figure(figsize=(7, 5))
plt.imshow(transition_matrix.to_numpy(), cmap="Blues", vmin=0, vmax=1)
plt.colorbar(label="Transition probability")
plt.xticks(range(len(states)), states)
plt.yticks(range(len(states)), states)
plt.xlabel("Next state")
plt.ylabel("Current state")
plt.title("Markov Chain Transition Probability Matrix")

for i in range(len(states)):
    for j in range(len(states)):
        plt.text(
            j, i, f"{transition_matrix.iloc[i, j]:.2f}",
            ha="center", va="center"
        )

plt.tight_layout()
plt.show()

# 7. Visualize multi-step probabilities
results_df = pd.DataFrame(
    multi_step_results,
    index=[f"Step {i}" for i in range(1, 6)],
    columns=states
)
results_df.plot(kind="line", marker="o", figsize=(8, 5))
plt.ylabel("Probability")
plt.xlabel("Future step")
plt.title("Multi-step State Probabilities Starting from Sunny")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Demonstrate the Markov property
print("\nMarkov property:")
print(
    "The next state is modeled using only the current state, "
    "not the full history of earlier states."
)
