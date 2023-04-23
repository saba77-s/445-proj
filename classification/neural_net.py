from import_data import X_train, X_test, y_train, y_test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
import torch
import torch.nn as nn

# Building Classifier
class NeuralNetClassifier(nn.Module):
    def __init__(
        self,
        hidden_layer_sizes,
        X=None,
        y=None,
        max_iter=30_000,
        learning_rate=0.1,
        init_scale=1,
        batch_size=1,
        weight_decay=1e-3,
        device=None,
    ):
        super().__init__()
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

        if X is not None and y is not None:
            self.fit(X,y)
    
    def tensorize(self, ary):
        return torch.as_tensor(ary, dtype=torch.get_default_dtype(), device=self.device)

    def build(self, in_dim, out_dim):
        layers = []
        lin = nn.Linear(in_dim, 100, device=self.device)
        nn.init.normal_(lin.weight, mean=0, std=self.init_scale)
        layers.append(lin)
        layers.append(nn.Tanh())
        lin = nn.Linear(100, 50, device=self.device)
        nn.init.normal_(lin.weight, mean=0, std=self.init_scale)
        layers.append(lin)
        layers.append(nn.Tanh())
        lin = nn.Linear(50, 3, device=self.device)
        nn.init.normal_(lin.weight, mean=0, std=self.init_scale)
        layers.append(lin)
        layers.append(nn.Tanh())       
        lin = nn.Linear(3, out_dim, device=self.device)
        nn.init.normal_(lin.weight, mean=0, std=self.init_scale)
        layers.append(lin)

        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers.forward(self.tensorize(x))
    
    def fit(self, X, y):
        X = self.tensorize(X)
        y = self.tensorize(y)
        if y.ndim == 1:
            y = y[:, np.newaxis]
        self.build(X.shape[1], y.shape[1])
        print(self.layers)
        loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for i in range(self.max_iter):
            self.optimizer.zero_grad()
            indices = torch.as_tensor(np.random.choice(X.shape[0], size=self.batch_size, replace=False))
            yhat = self(X[indices])
            loss = loss_fn(yhat, y[indices])
            loss.backward()
            self.optimizer.step()

            if i % 500 == 0:
                print(f"Iteration {i:>10,}: loss = {loss:>6.3f}")
    
    def predict(self, X):
        with torch.no_grad():
            f = nn.Sigmoid()
            preds = f(self(X)).cpu().numpy()
            if preds.shape[1] == 1:
                return (np.squeeze(preds, 1) < 0.5).astype(int)
            else:
                return np.argmax(preds,1)

model = NeuralNetClassifier([3], device="cpu")
model.fit(X_train, y_train)
yhat = model.predict(X_test)

print(yhat[0:10])
print("test error: " + str(np.mean(yhat != y_test)))