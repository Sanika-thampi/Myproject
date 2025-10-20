import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import pickle

# Load updated dataset
df = pd.read_csv("retail_store_inventory.csv")

# Fix date parsing
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=["Date"])

# Sort properly
df = df.sort_values(["Store ID", "Product ID", "Date"])

models = {}

# Train Holt-Winters model per store-product combo
for (store, product), group in df.groupby(["Store ID", "Product ID"]):
    group = group.set_index("Date").asfreq("D").fillna(method="ffill")
    if "Units Sold" not in group.columns:
        continue

    try:
        model = ExponentialSmoothing(
            group["Units Sold"],
            trend="add",
            seasonal="add",
            seasonal_periods=7
        ).fit()
        models[(store, product)] = model
        print(f"✅ Trained model for {store} - {product}")
    except Exception as e:
        print(f"⚠️ Skipped {store}-{product}: {e}")

# Save all models
with open("holt_winters_models.pkl", "wb") as f:
    pickle.dump(models, f)

print("✅ All Holt-Winters models saved successfully!")
