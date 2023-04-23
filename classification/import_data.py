import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv("Breast_GSE70947.csv")
data = data.to_numpy()
y = data[:,1]
y = 1 * (y == 'normal')
X = data[:,2:]

# splitting the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)

np.savetxt("X_train.csv", X_train, delimiter=",")
np.savetxt("X_test.csv", X_test, delimiter=",")
np.savetxt("y_train.csv", y_train, delimiter=",")
np.savetxt("y_test.csv", y_test, delimiter=",")

print(np.unique(y_train, return_counts=True)[1]/y_train.shape[0])
print(np.unique(y_test, return_counts=True)[1]/y_test.shape[0])
print(np.unique(y, return_counts=True)[1]/y.shape[0])

print(X_train[0:3,:])