from base import *
import numpy as np

# TODO: Perform the experiments and write this part. Report the outcomes (10 averages over 20
# repeats, and standard deviations of these 20 values) in a table and also in a figure/graph/plot.
# Explain what you observe in the table and/or figure, and answer research question 1 as far as
# possible. Use at most one page, including table and figure.

n_values = np.arange(100, 1001, 100)
y_values = np.zeros(n_values.size)

for i in range(n_values.size):
	n = n_values[i]
	y_values[i] = run_experiment_batch(n, 10000, 7, 0, 0)

plt.title("Misclassifications and Density")
plt.xlabel("Number of synthetic points")
plt.ylabel("Number of misclassifications")
plt.xticks(n_values)
plt.plot(n_values, y_values)

# To visualise single experiment:
result = run_experiment_sample(1000, 10000, 7, 0.5, 0.1)
plot_points(result.S, result.s_labels)
plt.show()