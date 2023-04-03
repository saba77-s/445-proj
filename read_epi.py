import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("Brain_GSE50161.csv")

n = data.type.value_counts().to_frame()
if __name__ == '__main__':
    print(n)