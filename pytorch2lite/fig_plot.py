import numpy as np, matplotlib.pyplot as plt, matplotlib.cm as cm, pylab

nseries = 10
colors = cm.rainbow(np.linspace(0, 1, nseries))

all_x = []
all_y = []
for i in range(nseries):
    x = np.random.random(12)+i/10.0
    y = np.random.random(12)+i/5.0
    plt.scatter(x, y, color=colors[i])
    all_x.extend(x)
    all_y.extend(y)

# Could I somehow do the next part (add identity_line) if I haven't been keeping track of all the x and y values I've seen?
identity_line = np.linspace(max(min(all_x), min(all_y)), min(max(all_x), max(all_y)))
plt.plot(identity_line, identity_line, color="black", linestyle="dashed", linewidth=3.0)

plt.show()