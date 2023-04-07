from scipy.stats import binom
import matplotlib.pyplot as plt
from read_epi import data

row, column = data.shape
# probability of having a cancer cell
p = 0.5
# defining the list of r values
r_values = list(range(row + 1))
# obtaining the mean and variance
mean, var = binom.stats(row, p)
# list of pmf values
dist = [binom.pmf(r, row, p) for r in r_values ]
# printing the table

print("r\tp(r)")
for i in range(row + 1):
    print(str(r_values[i]) + "\t" + str(dist[i]))
# printing mean and variance
print("mean = "+str(mean))
print("variance = "+str(var))

plt.bar(r_values, dist)
plt.show()

print(data)
print(row, column)
