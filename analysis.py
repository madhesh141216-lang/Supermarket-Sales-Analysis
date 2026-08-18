import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

df = pd.read_csv("dataset/supermarket_sales.csv")
print(df.head())

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values("Date").reset_index(drop=True)

df["Lag_7"] = df["Sales"].shift(7)
df["Lag_14"] = df["Sales"].shift(14)

df["Rolling_7"] = df["Sales"].rolling(window=7).mean()
df["Rolling_14"] = df["Sales"].rolling(window=14).mean()

df_model = df.dropna().copy()

features = ["Lag_7", "Lag_14", "Rolling_7", "Rolling_14"]

X = df_model[features]
y = df_model["Sales"]

# Gradient Boosting Regressor

model = GradientBoostingRegressor(
    n_estimators=70,
    learning_rate=0.5,
    max_depth=3,
    random_state=42
)

model.fit(X, y)

print("=" * 70)
print("Gradient Boosting Model trained successfully!")

# Last available date
last_date = df["Date"].max()

# Future 7 dates
future_dates = pd.date_range(
    start=last_date + pd.Timedelta(days=1),
    periods=7,
    freq="D"
)

print("=" * 70)
print("Future Dates:")
print(future_dates)

# Copy historical sales for future prediction
sales_history = df["Sales"].tolist()

future_predictions = []

print("=" * 70)
print("Starting Future Sales Prediction...")

# Predict next 7 days one by one

for date in future_dates:

    lag_7 = sales_history[-7]
    lag_14 = sales_history[-14]

    rolling_7 = sum(sales_history[-7:]) / 7
    rolling_14 = sum(sales_history[-14:]) / 14

    future_features = pd.DataFrame([{
        "Lag_7": lag_7,
        "Lag_14": lag_14,
        "Rolling_7": rolling_7,
        "Rolling_14": rolling_14
    }])

    prediction = model.predict(future_features)[0]

    future_predictions.append(prediction)

    # Add prediction to history for next day's prediction
    sales_history.append(prediction)

    print(date.date(), "→", round(prediction, 2))

# Create Future Sales Prediction DataFrame

future_sales = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_predictions
})

future_sales["Predicted_Sales"] = future_sales["Predicted_Sales"].round(2)

print("=" * 70)
print("Future 7 Days Sales Prediction:")
print(future_sales)


# Future 7 Days Sales Prediction Graph

plt.figure(figsize=(10, 5))

plt.plot(
    future_sales["Date"],
    future_sales["Predicted_Sales"],
    marker="o"
)

plt.title("Future 7 Days Sales Prediction")
plt.xlabel("Date")
plt.ylabel("Predicted Sales")

plt.xticks(rotation=45, ha="right")

plt.tight_layout()

plt.savefig("static/images/future_sales_prediction.png")

plt.close()

print("=" * 70)
print("Future Sales Prediction graph saved successfully!")


# Recent Actual Sales + Future Predicted Sales

recent_sales = df.tail(30)

plt.figure(figsize=(12, 6))

plt.plot(
    recent_sales["Date"],
    recent_sales["Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    future_sales["Date"],
    future_sales["Predicted_Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Recent Actual Sales vs Future 7 Days Prediction")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45, ha="right")
plt.legend()

plt.tight_layout()

plt.savefig("static/images/actual_vs_future_prediction.png")

plt.close()

print("=" * 70)
print("Actual vs Future Prediction graph saved successfully!")

print("=" * 70)
print("Features:")
print(X.head())

print("=" * 70)
print("Target:")
print(y.head())

print(df_model[["Date", "Sales", "Lag_7", "Lag_14", "Rolling_7", "Rolling_14"]].head())

print("=" * 70)
print("Shape:")
print(df.shape)

print("=" * 70)
print("Columns:")
print(df.columns)

print("=" * 70)
df.info()

print("=" * 70)
print(df.describe())

print("=" * 70)
print(df.isnull().sum())

print("=" * 70)
print(df.duplicated().sum())

print("=" * 70)
sales_by_product = df.groupby("Product line")["Sales"].sum()
sales_by_product = sales_by_product.sort_values(ascending=False)
print(sales_by_product)

sales_by_product.plot(kind="bar")

plt.title("Sales by Product")
plt.xlabel("Product Line")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("static/images/sales_by_product.png")
plt.close()

print("=" * 70)
sales_by_branch = df.groupby("Branch")["Sales"].sum()
sales_by_branch = sales_by_branch.sort_values(ascending=False)
print(sales_by_branch)

sales_by_branch.plot(kind="bar")

plt.title("Sales By Branch")
plt.xlabel("Branch")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("static/images/sales_by_branch.png")
plt.close()

print("=" * 70)
sales_by_payment = df.groupby("Payment")["Sales"].sum()
sales_by_payment = sales_by_payment.sort_values(ascending=False)
print(sales_by_payment)

sales_by_payment.plot(kind="pie", autopct="%1.1f%%")
 

plt.title("Sales by Payment")

plt.tight_layout()
plt.savefig("static/images/sales_by_payment.png")
plt.close()


print("=" * 70)
sales_by_customer_type = df.groupby("Customer type")["Sales"].sum()
sales_by_customer_type = sales_by_customer_type.sort_values(ascending=False)
print(sales_by_customer_type)

sales_by_customer_type.plot(kind="bar")

plt.title("Sales By Customer Type")
plt.xlabel("Customer")
plt.ylabel("Total Sales")

plt.tight_layout()
plt.savefig("static/images/sales_by_customer_type.png")
plt.close()


print("=" *70)

sales_by_city = df.groupby("City")["Sales"].sum()
sales_by_city = sales_by_city.sort_values(ascending=False)
print( sales_by_city)

sales_by_city.plot(kind="bar")

plt.title("Sales By City")
plt.xlabel("City")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("static/images/sales_by_city.png")
plt.close()


print("=" *70)
sales_by_gender = df.groupby("Gender")["Sales"].sum()
sales_by_gender = sales_by_gender.sort_values(ascending=False)
print(sales_by_gender )

sales_by_gender.plot(kind="bar")

plt.title("Sales By Gender")
plt.xlabel("Gender")
plt.ylabel("Sales")

plt.tight_layout()
plt.savefig("static/images/sales_by_gender.png")
plt.close()


print("=" *70)
rating_by_product = df.groupby("Product line")["Rating"].mean()
rating_by_product =rating_by_product.sort_values(ascending=False)
print(rating_by_product)

rating_by_product.plot(kind="bar")

plt.title("Average Rating by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Average Rating")

plt.tight_layout()
plt.savefig("static/images/rating_by_product.png")
plt.close()


print("=" * 100)
print("Highest Sale:")
print(df["Sales"].max())

print("=" * 100)
print("Lowest Sale:")
print(df["Sales"].min())

print("=" * 100)
payment_count = df["Payment"].value_counts()
print(payment_count)

payment_count.plot(kind="bar")

plt.title("Payment Method Count")
plt.xlabel("Payment Method")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("static/images/payment_count.png")
plt.close()


print("=" * 100)
gender_count = df["Gender"].value_counts()
print(gender_count)

gender_count.plot(kind="pie", autopct="%1.1f%%")
plt.title("Gender Distribution")

plt.tight_layout()
plt.savefig("static/images/gender_count.png")
plt.close()


print("=" * 100)
product_count = df["Product line"].value_counts()
print(product_count)

product_count.plot(kind="bar")

plt.title("Product Count")
plt.xlabel("Product Line")
plt.ylabel("Number of Transactions")



print("=" * 100)
high_sales = df[df["Sales"] > 500]
print(high_sales)

print("=" * 100)
female_customers = df[df["Gender"] == "Female"]
print(female_customers)

print("=" * 100)
cash_payment = df[df["Payment"] == "Cash"]
print(cash_payment)

print("=" * 100)
high_rating = df[df["Rating"] > 9]
print(high_rating )

print("=" * 100)
high_quality = df[df["Quantity"] >= 8]
print(high_quality)

print("=" *100)
female_high_sales = df[(df["Gender"] == "Female") & (df["Sales"] > 500)]
print(female_high_sales)


print("=" *100)
cash_high_rating = df[(df["Payment"] == "Cash") & (df["Rating"] > 8)]
print(cash_high_rating)

print("=" *100)
member_high_sales = df[(df["Customer type"] == "Member") & (df["Sales"] > 800)]
print(member_high_sales)

print("=" *100)
female_or_cash = df[(df["Gender"] == "Female") | (df["Payment"] == "Cash")]  
print(female_or_cash)


print("=" *100)
df["Date"] = pd.to_datetime(df["Date"])
print(df["Date"].head())
print(df["Date"].dtype)

df['Month'] = df['Date'].dt.to_period('M')
monthly_sales = df.groupby('Month')['Sales'].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(kind='line', marker='o')

plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("static/images/monthly_sales.png")
plt.close()

print("=" *100)
daily_sales = df.groupby("Date")["Sales"].sum()
print(daily_sales)

plt.figure(figsize=(10,5))
daily_sales.plot()
plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.savefig("static/images/daily_sales_trend.png")
plt.close()

print("=" *100)
quantity_by_product = df.groupby("Product line")["Quantity"].sum()
quantity_by_product = quantity_by_product.sort_values(ascending=False)
print(quantity_by_product)

plt.figure(figsize=(12,6))

quantity_by_product.plot(kind='bar')

plt.title("Quantity Sold by Product Line")
plt.xlabel("Product Line")
plt.ylabel("Quantity")

plt.xticks(rotation=45, ha="right", fontsize=10)

plt.subplots_adjust(bottom=0.30)

plt.savefig(
    "static/images/quantity_by_product_line.png",
    bbox_inches="tight"
)

plt.close()


print("=" *100)
correlation = df.corr(numeric_only=True)
plt.figure(figsize=(10,6))
plt.imshow(correlation)
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Correlation Analysis")
plt.tight_layout()
plt.savefig("static/images/correlation.png")
plt.close()


print("=" * 100)
top5_products = df.groupby("Product line")["Sales"].sum()
top5_products = top5_products.sort_values(ascending=False).head(5)
print(top5_products)

plt.figure(figsize=(8,5))
top5_products.plot(kind="barh")

plt.title("Top 5 Products by Sales")
plt.xlabel("Total Sales")
plt.ylabel("Product Line")
plt.tight_layout()
plt.savefig("static/images/top5_products.png")
plt.close()