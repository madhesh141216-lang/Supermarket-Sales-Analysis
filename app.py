from flask import Flask, render_template, request, Response, redirect, url_for
import io
import glob
import os
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.ensemble import GradientBoostingRegressor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/upload", methods=["POST"])
def upload_file():
    global df, branches, cities

    file = request.files.get("file")

    if not file or file.filename == "":
        return redirect(
            url_for(
                "home",
                upload_error="Please select a CSV file."
            )
        )

    if not file.filename.lower().endswith(".csv"):
        return redirect(
            url_for(
                "home",
                upload_error="Only CSV files are allowed."
            )
        )

    try:
        uploaded_df = pd.read_csv(file)

        # Clean column names
        uploaded_df.columns = uploaded_df.columns.str.strip()

        # Apply standard column mapping
        uploaded_df.rename(
            columns=column_mapping,
            inplace=True
        )

        # Check required columns
        missing_columns = [
            column
            for column in required_columns
            if column not in uploaded_df.columns
        ]

        if missing_columns:
            return redirect(
                url_for(
                    "home",
                    upload_error="Missing columns: "
                    + ", ".join(missing_columns)
                )
            )

        # Convert Date column
        uploaded_df["Date"] = pd.to_datetime(
            uploaded_df["Date"],
            errors="coerce"
        )

        uploaded_df = uploaded_df.dropna(
            subset=["Date"]
        )

        if uploaded_df.empty:
            return redirect(
                url_for(
                    "home",
                    upload_error="CSV does not contain valid Date values."
                )
            )

        # Save uploaded CSV
        filename = secure_filename(file.filename)
        upload_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        uploaded_df.to_csv(
            upload_path,
            index=False
        )

        # Update current dataframe
        df = uploaded_df

        # Update filters
        branches = sorted(
            df["Branch"].dropna().unique()
        )

        cities = sorted(
            df["City"].dropna().unique()
        )

        print("CSV uploaded successfully!")
        print("Rows:", len(df))
        print("Uploaded file:", upload_path)

        return redirect(
            url_for(
                "home",
                upload_success=
                f"CSV uploaded successfully! {len(df)} rows loaded."
            )
        )

    except Exception as e:

        print("CSV Upload Error:", e)

        return redirect(
            url_for(
                "home",
                upload_error=f"Error reading CSV: {str(e)}"
            )
        )
    
# Automatically find CSV dataset
csv_files = glob.glob(os.path.join("dataset", "*.csv"))

if not csv_files:
    raise FileNotFoundError("No CSV dataset found inside the dataset folder.")

dataset_path = csv_files[0]

df = pd.read_csv(dataset_path)

# Clean column names
df.columns = df.columns.str.strip()

# Standard column name mapping
column_mapping = {
    "Revenue": "Sales",
    "Total Sales": "Sales",
    "Sales Amount": "Sales",
    "Units Sold": "Quantity",
    "Units": "Quantity",
    "Product": "Product line",
    "Product Line": "Product line",
    "Store": "Branch",
    "Store Branch": "Branch",
    "Location": "City",
    "Customer Rating": "Rating",
    "Rating Score": "Rating",
    "Payment Method": "Payment",
    "Customer Type": "Customer type"
}

df.rename(columns=column_mapping, inplace=True)

print("Dataset columns:", list(df.columns))

# Required columns for this dashboard
required_columns = [
    "Sales",
    "Quantity",
    "Product line",
    "Branch",
    "City",
    "Rating",
    "Payment",
    "Gender",
    "Customer type",
    "Date"
]

print("Dataset columns:", list(df.columns))

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Dataset is missing required columns: "
        + ", ".join(missing_columns)
    )

branches = sorted(df["Branch"].dropna().unique())
cities = sorted(df["City"].dropna().unique())

print("Dataset loaded:", dataset_path)
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(app.static_folder)


def load_selected_dataset():

    selected_dataset = request.args.get(
        "dataset",
        "default"
    )

    # Default dataset
    if selected_dataset == "default":
        return df

    # Uploaded dataset
    upload_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        selected_dataset
    )

    if os.path.exists(upload_path):

        selected_df = pd.read_csv(upload_path)

        selected_df.columns = selected_df.columns.str.strip()

        selected_df["Date"] = pd.to_datetime(
            selected_df["Date"],
            errors="coerce"
        )

        return selected_df

    # If file not found
    return df

def create_chart(data, title, xlabel="", chart_type="bar"):
    fig, ax = plt.subplots(figsize=(8, 4))

    if chart_type == "pie":
        data.plot(
            kind="pie",
            autopct="%1.1f%%",
            startangle=90,
            ax=ax
        )
        ax.set_ylabel("")
    else:
        data.plot(
            kind="bar",
            ax=ax
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Sales")
        ax.tick_params(axis="x", rotation=45)

    ax.set_title(title)
    fig.tight_layout()

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=80)
    plt.close(fig)

    img.seek(0)

    return img

def generate_future_predictions(data):

    prediction_df = data.copy()

    prediction_df["Date"] = pd.to_datetime(
        prediction_df["Date"],
        errors="coerce"
    )

    prediction_df = prediction_df.dropna(
        subset=["Date", "Sales"]
    )

    prediction_df = (
        prediction_df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Lag features
    prediction_df["Lag_7"] = prediction_df["Sales"].shift(7)
    prediction_df["Lag_14"] = prediction_df["Sales"].shift(14)

    # Rolling features
    prediction_df["Rolling_7"] = (
        prediction_df["Sales"].shift(1).rolling(7).mean()
    )

    prediction_df["Rolling_14"] = (
        prediction_df["Sales"].shift(1).rolling(14).mean()
    )

    prediction_df = prediction_df.dropna().copy()

    features = [
        "Lag_7",
        "Lag_14",
        "Rolling_7",
        "Rolling_14"
    ]

    X = prediction_df[features]
    y = prediction_df["Sales"]

    model = GradientBoostingRegressor(
        n_estimators=70,
        learning_rate=0.5,
        max_depth=3,
        random_state=42
    )

    model.fit(X, y)

    last_date = prediction_df["Date"].max()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=7,
        freq="D"
    )

    sales_history = prediction_df["Sales"].tolist()

    future_predictions = []

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

        prediction = model.predict(
            future_features
        )[0]

        future_predictions.append(prediction)

        sales_history.append(prediction)

    future_sales = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Sales": future_predictions
    })

    future_sales["Predicted_Sales"] = (
        future_sales["Predicted_Sales"].round(2)
    )

    return future_sales

@app.route("/")
def home():

    global df, branches, cities

    # Get uploaded CSV files
    upload_files = sorted([
        file
        for file in os.listdir(app.config["UPLOAD_FOLDER"])
        if file.lower().endswith(".csv")
    ])

    # Selected dataset
    selected_dataset = request.args.get(
        "dataset",
        "default"
    )

    # Load selected dataset
    if selected_dataset != "default":

        selected_path = os.path.join(
            app.config["UPLOAD_FOLDER"],
            selected_dataset
        )

        if os.path.exists(selected_path):

            df = pd.read_csv(selected_path)

            df.columns = df.columns.str.strip()

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

            branches = sorted(
                df["Branch"].dropna().unique()
            )

            cities = sorted(
                df["City"].dropna().unique()
            )

    # Branch and City filters
    selected_branch = request.args.get(
        "branch",
        "All"
    )

    selected_city = request.args.get(
        "city",
        "All"
    )

    future_sales = generate_future_predictions(df)

    filtered_df = load_selected_dataset()

    if selected_branch != "All":

        available_cities = sorted(
            df[
                df["Branch"] == selected_branch
            ]["City"].unique()
        )

    else:

        available_cities = cities

    if selected_branch != "All":

        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":

        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    if filtered_df.empty:

        filtered_df = df

    # KPI calculations
    total_sales = round(
        filtered_df["Sales"].sum(),
        2
    )

    total_sales = f"{total_sales:,.2f}"

    total_transactions = len(filtered_df)

    average_rating = round(
        filtered_df["Rating"].mean(),
        2
    )

    total_branches = filtered_df["Branch"].nunique()

    # Sales by Product
    sales_by_product = (
        filtered_df
        .groupby("Product line")["Sales"]
        .sum()
    )

    highest_product = sales_by_product.idxmax()

    product_labels = sales_by_product.index.tolist()

    product_values = sales_by_product.values.tolist()

    # Sales by Branch
    sales_by_branch = (
        filtered_df
        .groupby("Branch")["Sales"]
        .sum()
    )

    highest_branch = sales_by_branch.idxmax()

    # Sales by Payment
    sales_by_payment = (
        filtered_df
        .groupby("Payment")["Sales"]
        .sum()
    )

    popular_payment = sales_by_payment.idxmax()

    # Sales by City
    sales_by_city = (
        filtered_df
        .groupby("City")["Sales"]
        .sum()
    )

    highest_city = sales_by_city.idxmax()

    # Highest and Lowest Sale
    highest_sale = filtered_df["Sales"].max()

    lowest_sale = filtered_df["Sales"].min()

    highest_sale = f"{highest_sale:,.2f}"

    lowest_sale = f"{lowest_sale:,.2f}"

    # Product analysis
    product_analysis = (
        filtered_df
        .groupby("Product line")
        .agg({
            "Sales": "sum",
            "Quantity": "sum",
            "Rating": "mean"
        })
    )

    best_sales_product = (
        product_analysis["Sales"].idxmax()
    )

    best_quantity_product = (
        product_analysis["Quantity"].idxmax()
    )

    best_rating_product = (
        product_analysis["Rating"].idxmax()
    )

    highest_sales_value = (
        product_analysis["Sales"].max()
    )

    highest_quantity_value = (
        product_analysis["Quantity"].max()
    )

    highest_rating_value = (
        product_analysis["Rating"].max()
    )

    # AI Insights
    sales_insight = (
        f"{best_sales_product} generated the highest sales "
        f"with ₹{highest_sales_value:,.2f}."
    )

    quantity_insight = (
        f"{best_quantity_product} had the highest quantity sold "
        f"with {int(highest_quantity_value)} units."
    )

    rating_insight = (
        f"{best_rating_product} received the highest average rating "
        f"of {highest_rating_value:.2f}."
    )

    return render_template(
        "dashboard.html",

        total_sales=total_sales,

        total_transactions=total_transactions,

        average_rating=average_rating,

        total_branches=total_branches,

        branches=branches,

        cities=available_cities,

        highest_product=highest_product,

        highest_branch=highest_branch,

        popular_payment=popular_payment,

        highest_city=highest_city,

        highest_sale=highest_sale,

        lowest_sale=lowest_sale,

        product_labels=product_labels,

        product_values=product_values,

        best_sales_product=best_sales_product,

        best_quantity_product=best_quantity_product,

        best_rating_product=best_rating_product,

        sales_insight=sales_insight,

        quantity_insight=quantity_insight,

        rating_insight=rating_insight,

        future_sales=future_sales.to_dict("records"),

        upload_files=upload_files,

        selected_dataset=selected_dataset
    )

@app.route("/chart/sales-product")
def sales_product_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_product = filtered_df.groupby("Product line")["Sales"].sum()

    img = create_chart(
        sales_by_product,
        "Sales by Product Line",
        "Product Line",
        "bar"
    )

    return Response(img.getvalue(), mimetype="image/png")


@app.route("/chart/sales-gender")
def sales_gender_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_gender = filtered_df.groupby("Gender")["Sales"].sum()

    img = create_chart(
        sales_by_gender,
        "Sales by Gender",
        "Gender",
        "pie"
    )

    return Response(img.getvalue(), mimetype="image/png")



@app.route("/chart/sales-branch")
def sales_branch_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_branch = filtered_df.groupby("Branch")["Sales"].sum()

    img = create_chart(
        sales_by_branch,
        "Sales by Branch"
    )

    return Response(img.getvalue(), mimetype="image/png")


@app.route("/chart/sales-city")
def sales_city_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_city = filtered_df.groupby("City")["Sales"].sum()

    img = create_chart(
        sales_by_city,
        "Sales by City"
    )

    return Response(img.getvalue(), mimetype="image/png")


@app.route("/chart/sales-customer-type")
def sales_customer_type_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_customer_type = filtered_df.groupby("Customer type")["Sales"].sum()

    img = create_chart(
        sales_by_customer_type,
        "Sales by Customer Type"
    )

    return Response(img.getvalue(), mimetype="image/png")

 
@app.route("/chart/rating-product")
def rating_product_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    rating_by_product = filtered_df.groupby("Product line")["Rating"].mean()

    img = create_chart(
        rating_by_product,
        "Average Rating by Product"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/quantity-product")
def quantity_product_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    quantity_by_product = filtered_df.groupby("Product line")["Quantity"].sum()

    img = create_chart(
        quantity_by_product,
        "Quantity Sold by Product Line"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/payment-count")
def payment_count_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    payment_count = filtered_df["Payment"].value_counts()

    img = create_chart(
        payment_count,
        "Payment Method Count"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/gender-count")
def gender_count_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    gender_count = filtered_df["Gender"].value_counts()

    img = create_chart(
        gender_count,
        "Gender Distribution"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/top5-products")
def top5_products_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    top5_products = (
        filtered_df.groupby("Product line")["Sales"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    img = create_chart(
        top5_products,
        "Top 5 Products by Sales"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/daily-sales")
def daily_sales_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()
    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    filtered_df = filtered_df.copy()
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"])

    daily_sales = filtered_df.groupby("Date")["Sales"].sum()

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        daily_sales.index,
        daily_sales.values,
        marker="o"
    )

    ax.set_title("Daily Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=80)
    plt.close(fig)

    img.seek(0)

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/correlation")
def correlation_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    correlation = filtered_df[
    ["Unit price", "Quantity", "Tax 5%", "Sales", "Rating"]
    ].corr()

    plt.figure(figsize=(8, 6))

    plt.imshow(correlation, cmap="coolwarm", interpolation="nearest")

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

    plt.colorbar()

    plt.title("Correlation Analysis")
    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=80)
    plt.close()

    img.seek(0)

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/monthly-sales")
def monthly_sales_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
            
        ]
    filtered_df = filtered_df.copy()
    filtered_df["Date"] = pd.to_datetime(filtered_df["Date"],errors="coerce")
    

    monthly_sales = (
        filtered_df
        .groupby(filtered_df["Date"].dt.to_period("M"))["Sales"]
        .sum()
    )

    monthly_sales.index = monthly_sales.index.astype(str)

    img = create_chart(
        monthly_sales,
        "Monthly Sales Trend"
    )

    return Response(img.getvalue(), mimetype="image/png")


@app.route("/chart/sales-payment")
def sales_payment_chart():
    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    sales_by_payment = filtered_df.groupby("Payment")["Sales"].sum()

    img = create_chart(
        sales_by_payment,
        "Sales by Payment",
        "Payment",
        "pie"
    )

    return Response(img.getvalue(), mimetype="image/png")

@app.route("/chart/customer-type-sales")
def customer_type_sales_chart():

    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    customer_sales = (
        filtered_df.groupby("Customer type")["Sales"]
        .sum()
    )

    img = create_chart(
        customer_sales,
        "Customer Type Sales Comparison",
        "Customer Type",
        "bar"
    )

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )


@app.route("/chart/customer-type-quantity")
def customer_type_quantity_chart():

    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    customer_quantity = (
        filtered_df.groupby("Customer type")["Quantity"]
        .sum()
    )

    img = create_chart(
        customer_quantity,
        "Customer Type Quantity Comparison",
        "Customer Type",
        "bar"
    )

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )

@app.route("/chart/customer-gender-sales")
def customer_gender_sales_chart():

    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()
         
    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    customer_gender_sales = (
        filtered_df
        .groupby(["Customer type", "Gender"])["Sales"]
        .sum()
        .unstack()
    )

    plt.figure(figsize=(8, 5))

    customer_gender_sales.plot(
        kind="bar",
        ax=plt.gca()
    )

    plt.title("Sales by Customer Type and Gender")
    plt.xlabel("Customer Type")
    plt.ylabel("Sales")
    plt.xticks(rotation=0)
    plt.legend(title="Gender")

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=80)
    plt.close()

    img.seek(0)

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )



@app.route("/chart/sales-vs-quantity")
def sales_vs_quantity_chart():

    selected_branch = request.args.get("branch", "All")
    selected_city = request.args.get("city", "All")

    filtered_df = load_selected_dataset()

    if selected_branch != "All":
        filtered_df = filtered_df[
            filtered_df["Branch"] == selected_branch
        ]

    if selected_city != "All":
        filtered_df = filtered_df[
            filtered_df["City"] == selected_city
        ]

    product_data = (
        filtered_df
        .groupby("Product line")
        .agg({
            "Quantity": "sum",
            "Sales": "sum"
        })
    )

    plt.figure(figsize=(9, 5))

    plt.scatter(
        product_data["Quantity"],
        product_data["Sales"]
    )

    for product in product_data.index:
        plt.annotate(
            product,
            (
                product_data.loc[product, "Quantity"],
                product_data.loc[product, "Sales"]
            )
        )

    plt.title("Sales vs Quantity by Product Line")
    plt.xlabel("Quantity Sold")
    plt.ylabel("Sales")

    plt.tight_layout()

    img = io.BytesIO()
    plt.savefig(img, format="png", dpi=80)
    plt.close()

    img.seek(0)

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )

@app.route("/chart/future-sales")
def future_sales_chart():

    future_sales = generate_future_predictions(df)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        future_sales["Date"],
        future_sales["Predicted_Sales"],
        marker="o"
    )

    ax.set_title("Future 7 Days Sales Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Sales")

    ax.tick_params(axis="x", rotation=45)

    fig.tight_layout()

    img = io.BytesIO()
    fig.savefig(img, format="png", dpi=80)
    plt.close(fig)

    img.seek(0)

    return Response(
        img.getvalue(),
        mimetype="image/png"
    )

if __name__ == "__main__":
    app.run(debug=True)