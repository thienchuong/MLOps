import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read dataset
df = pd.read_csv('data.csv')
df.columns = ["index", "height", "weight"]
print(df.head())


sns.scatterplot(
    data=df,
    x="height",
    y="weight",
    color="blue"
)
plt.show()

# get x and y
x = df["height"].values
y = df["weight"].values

# find m and b
N = x.shape[0]
m = (N * np.sum(x * y) - np.sum(x) * np.sum(y)) / (N * np.sum(x * x) - np.sum(x) * np.sum(x))
b = (np.sum(y) - m * np.sum(x)) / N
print(m,b)

# draw regresstion line
x_min = np.min(x)
y_min = m * x_min + b
x_max = np.max(x)
y_max = m * x_max + b

fig, ax = plt.subplots()
sns.scatterplot(
    data=df,
    x="height",
    y="weight",
    color="blue",
    ax=ax,
    alpha=0.4
)
sns.lineplot(
    x=[x_min, x_max],
    y=[y_min, y_max],
    color="red",
    ax=ax
)
plt.show()