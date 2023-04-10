from sklearn.mixture import BayesianGaussianMixture
import numpy as np
import pandas as pd

data = pd.read_csv("Brain_GSE50161.csv", usecols = range(2,16384))

# Extract the data as a numpy array
X = data.values

# Set the number of components (clusters) you want to model
n_components = 1

# Fit the Bayesian Gaussian Mixture model to the data
model = BayesianGaussianMixture(n_components=n_components, covariance_type='full').fit(X)

# Get the cluster assignments for each data point
labels = model.predict(X)

# Get the probability density for each data point
densities = np.exp(model.score_samples(X))

if __name__ == '__main__':
    # Print the parameters for each component
    for i in range(n_components):
        print('Component {}:'.format(i))
        print('  Mean:', model.means_[i])
        print('  Covariance Matrix:', model.covariances_[i])
        print('  Weight:', model.weights_[i])