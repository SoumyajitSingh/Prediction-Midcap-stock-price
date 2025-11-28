from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# ===== Load and Prepare Data =====
df = pd.read_csv("FAANG.csv")

features = [
    "Open", "High", "Low", "Volume", "Market Cap", "PE Ratio", "EPS", "Forward PE",
    "Net Income", "Debt to Equity", "Return on Equity (ROE)", "Current Ratio",
    "Free Cash Flow", "Operating Margin", "Profit Margin", "Quick Ratio",
    "Price to Book Ratio", "Enterprise Value", "Total Debt"
]

target = "Close"

df = df.dropna(subset=[target])
X = df[features]
y = df[target]

imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)

# ===== Routes =====
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            input_data = [float(request.form[feat]) for feat in features]
            input_data_imputed = imputer.transform([input_data])
            prediction = round(model.predict(input_data_imputed)[0], 2)
        except Exception as e:
            error = f"Invalid input: {e}"
    return render_template("index.html", features=features, prediction=prediction, error=error)

if __name__ == "__main__":
    app.run(debug=True)
