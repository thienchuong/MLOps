import pandas as pd
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df=pd.read_csv('data.csv')
df.columns = ["index", "height", "weight"]


x=df["height"].values
y=df["weight"].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train the Linear Regression Model
model = LinearRegression()
model.fit(x.reshape(-1, 1), y)
print(model.predict([[65]]))