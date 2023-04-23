import pandas as pd
X_train = pd.read_csv("X_train.csv").to_numpy()
X_test = pd.read_csv("X_test.csv").to_numpy()
y_train = pd.read_csv("y_train.csv").to_numpy()
y_test = pd.read_csv("y_test.csv").to_numpy()