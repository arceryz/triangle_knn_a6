from base import *

# TODO: Perform the experiments and write this part. Report the outcomes (7 averages over 20
# repeats, and standard deviations of these 20 numbers) in a table and also in a figure/graph/plot.
# Explain what you observe in the table and/or figure, and answer research question 2 as far as
# possible. Use at most one page, including table and figure.

f_values = np.arange(0.0, 0.35, 0.05, dtype=float)
y_values = np.zeros(f_values.size)
sigma_values = np.zeros(f_values.size)


for i in range(f_values.size):
	f = f_values[i]
	batch = run_experiment_batch(1000, 10000, 7, f, f)
	y_values[i] = batch.mean()
	sigma_values[i] = batch.std()

plt.title("Misclassifications and Outlier Fraction")
plt.xlabel("Probability of reversing colors")
plt.ylabel("Number of misclassifications")
plt.xticks(f_values)
plt.plot(f_values, y_values)

# To visualise single experiment:
result = run_experiment_sample(1000, 10000, 7, 0.3, 0.3)
plot_points(result.S, result.s_labels)
plt.show()