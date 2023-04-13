import warnings

from scipy.stats import nbinom
import matplotlib.pyplot as plt, seaborn as sns
import numpy as np
import statsmodels.api as sm
import pandas as pd
warnings.filterwarnings('ignore')

data = pd.read_csv("Brain_GSE50161.csv")
data = data.to_numpy()
row, column = data.shape
# probability of having a cancer cell
p = 0.2
def get_hist_and_indices(X, i):
    # this is just binning the original, continuous gene expression data into discrete bins (use these bins so that we are consistent!)
    sample = X[:, i+2]
    hist = np.histogram(sample, bins=15, density=False)
    indices = np.digitize(np.float64(sample), np.float64(hist[1]), right=False) # don't think we need this, just in case tho
    return sample, hist


sample, hist = get_hist_and_indices(data, 3)

print(sample.mean())
print(sample.var())

X = np.ones_like(sample)
X = np.array(X, dtype=float)
sample = np.array(sample, dtype=float)
res = sm.NegativeBinomial(sample, X).fit(start_params=[1,1])
print(res.summary)
mu = np.exp(res.params[0])
p = 1/(1+np.exp(res.params[0])*res.params[1])
n = np.exp(res.params[0])*p/(1-p)
x_plot = np.linspace(0,20,21)
sns.set_theme()
ax = sns.distplot(sample, kde=False, norm_hist=True, label='Real Values')
ax.plot(x_plot, nbinom.pmf(x_plot, n, p), 'g-', lw=2, label = 'fitted NB')
leg = ax.legend()
plt.title('real vs. Fitted NB distribution')
plt.show()