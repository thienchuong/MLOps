import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.dropna(subset=["height", "weight"])

# -------- Weight -> Height --------
X_w2h = df[["weight"]]
y_h = df["height"]
Xtr, Xte, ytr, yte = train_test_split(X_w2h, y_h, test_size=0.2, random_state=42)
m_w2h = LinearRegression().fit(Xtr, ytr)
pred_h = m_w2h.predict(Xte)
print(f"Weight->Height: R2={r2_score(yte, pred_h):.3f}, RMSE={mean_squared_error(yte, pred_h, squared=False):.3f}")
print(f"w->h coeff (slope)={m_w2h.coef_[0]:.4f}, intercept={m_w2h.intercept_:.4f}")

# -------- Height -> Weight --------
X_h2w = df[["height"]]
y_w = df["weight"]
Xtr, Xte, ytr, yte = train_test_split(X_h2w, y_w, test_size=0.2, random_state=42)
m_h2w = LinearRegression().fit(Xtr, ytr)
pred_w = m_h2w.predict(Xte)
print(f"Height->Weight: R2={r2_score(yte, pred_w):.3f}, RMSE={mean_squared_error(yte, pred_w, squared=False):.3f}")
print(f"h->w coeff (slope)={m_h2w.coef_[0]:.4f}, intercept={m_h2w.intercept_:.4f}")

# -------- Example predictions --------
w_val = 70  # kg
h_hat = m_w2h.predict(np.array([[w_val]]))[0]
print(f"Predicted height for {w_val} kg: {h_hat:.2f}")

h_val = 170  # cm
w_hat = m_h2w.predict(np.array([[h_val]]))[0]
print(f"Predicted weight for {h_val} cm: {w_hat:.2f}")