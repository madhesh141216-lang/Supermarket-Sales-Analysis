import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("dataset/supermarket_sales.csv")

df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

daily_sales = (
    df.groupby("Date")["Sales"]
    .sum()
    .reset_index()
)

daily_sales["Lag_7"] = daily_sales["Sales"].shift(7)
daily_sales["Lag_14"] = daily_sales["Sales"].shift(14)

daily_sales["Rolling_Mean_7"] = (
    daily_sales["Sales"].shift(1).rolling(7).mean()
)

daily_sales["Rolling_Mean_14"] = (
    daily_sales["Sales"].shift(1).rolling(14).mean()
)

daily_sales["Rolling_Mean_30"] = (
    daily_sales["Sales"].shift(1).rolling(30).mean()
)
 

daily_sales = daily_sales.dropna()

print(daily_sales.head())

daily_sales["Day"] = daily_sales["Date"].dt.day
daily_sales["Month"] = daily_sales["Date"].dt.month
daily_sales["Year"] = daily_sales["Date"].dt.year
daily_sales["DayOfWeek"] = daily_sales["Date"].dt.dayofweek

daily_sales["IsWeekend"] = (
    daily_sales["DayOfWeek"] >= 5
).astype(int)

X = daily_sales[[
    "Day",
    "Month",
    "Year",
    "DayOfWeek",
    "IsWeekend",
    "Lag_7",
    "Lag_14",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_30"
]]

y = daily_sales["Sales"]

split_index = int(len(X) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = GradientBoostingRegressor(
    n_estimators=70,
    learning_rate=0.5,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Absolute Error:", round(mae, 2))
print("R² Score:", round(r2, 4))

plt.figure(figsize=(10, 5))

plt.plot(y_test.values, label="Actual Sales")
plt.plot(y_pred, label="Predicted Sales")

plt.title("Actual vs Predicted Sales")
plt.xlabel("Test Data")
plt.ylabel("Sales")
plt.legend()

plt.tight_layout()
plt.show()

print("Mean Absolute Error:", round(mae, 2))

print(daily_sales[[
    "Date",
    "Sales",
    "Lag_7",
    "Lag_14",
    "Rolling_Mean_7",
    "Rolling_Mean_14",
    "Rolling_Mean_30"
]].head(20))

print("Training data:", len(X_train))
print("Testing data:", len(X_test))

print("Actual Sales:")
print(y_test.head())

print("Predicted Sales:")
print(y_pred[:5])