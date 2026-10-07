# region book:pricing-explainability-shap
import numpy as np
import pandas as pd

try:
    import shap
except ModuleNotFoundError:
    raise SystemExit("Install the explainability extra: uv sync --extra explainability") from None
from sklearn.ensemble import RandomForestRegressor

# Sample training data for a pricing model (for illustration purposes)
data = pd.DataFrame(
    {
        "inventory_level": [200, 50, 120, 80, 300],  # units in stock
        "competitor_price": [50, 45, 60, 55, 40],  # competitor's price in $
        "days_to_season_end": [10, 5, 30, 20, 15],  # days until end-of-season
    }
)
target = np.array([45, 40, 60, 50, 35])  # historical optimal prices for those scenarios

# Train a simple model (Random Forest) to predict optimal price
model = RandomForestRegressor(random_state=0).fit(data, target)

# Suppose the agent suggests a new price for a product with the following features:
product = pd.DataFrame(
    {
        "inventory_level": [150],  # current stock
        "competitor_price": [48],  # competitor's price for similar item
        "days_to_season_end": [7],  # days left in season
    }
)

predicted_price = model.predict(product)[0]
print(f"AI-predicted optimal price: ${predicted_price:.2f}")

# Use SHAP to explain the prediction
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(product)

# Pair each feature with its SHAP contribution value
explanation = {}
for feature_name, value, shap_val in zip(product.columns, product.iloc[0], shap_values[0], strict=False):
    explanation[feature_name] = round(shap_val, 2)
    print(f"  {feature_name}: {value} -> contribution {shap_val:+.2f}")

# The explanation dict now holds feature contributions to the price prediction.
# endregion book:pricing-explainability-shap
