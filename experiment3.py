from base import *

# TODO: Perform the experiments and write this part, and answer research question 3. Use at most
# one-and-a-half page, including table and figure.

f = 0.15
x_values = np.linspace(0, 1, 100)

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, title, params in [
	(axes[0], "$f_{in}=0.15$, $f_{out}=x f$", lambda x: (f, x * f)),
	(axes[1], "$f_{out}=0.15$, $f_{in}=x f$", lambda x: (x * f, f)),
]:
	red, blue = np.zeros(x_values.size), np.zeros(x_values.size)
	for i, x in enumerate(x_values):
		fi, fo = params(x)
		runs = [run_experiment_sample(1000, 10000, 7, fi, fo) for _ in range(30)]
		red[i] = np.mean([r.num_false_red for r in runs])
		blue[i] = np.mean([r.num_false_blue for r in runs])
	ax.plot(x_values, red + blue, label="Total", color="black")
	ax.plot(x_values, red, label="False red (inside)", color="red", linestyle="--")
	ax.plot(x_values, blue, label="False blue (outside)", color="blue", linestyle="-.")
	ax.set_title(title)
	ax.set_xlabel("x")
	ax.legend()

axes[0].set_ylabel("Number of misclassifications")
plt.show()
