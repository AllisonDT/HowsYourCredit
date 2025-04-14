import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformer_model import TabularTransformer
from sklearn.preprocessing import StandardScaler

# --------------------------
# Load test data
# --------------------------
df = pd.read_csv("test.csv")
ids = df["ID"].tolist()  # ✅ Grab real IDs before dropping

# Drop columns not used in training
df = df.drop(columns=[
    'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Monthly_Balance'
], errors='ignore')

# Identify numeric & categorical columns
categorical_cols = df.select_dtypes(include='object').columns.tolist()
numeric_cols = df.select_dtypes(include='number').columns.tolist()

# Encode categoricals
for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes

# Scale numerics
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Final input array
X_test = df.astype(np.float32).to_numpy()

# --------------------------
# Load model
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_test.shape[1]

model = TabularTransformer(input_dim=input_dim)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# --------------------------
# Run inference
# --------------------------
with torch.no_grad():
    X_tensor = torch.tensor(X_test).to(device)
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

# --------------------------
# Map predictions to labels
# --------------------------
label_map = {
    0: "Poor",
    1: "Standard",
    2: "Good"
}
pred_labels = [label_map[p] for p in preds]

# --------------------------
# Save to predictions_transformer.csv
# --------------------------
df_out = pd.DataFrame({
    "ID": ids,
    "Predicted_Credit_Score": pred_labels
})
df_out.to_csv("predictions_transformer.csv", index=False)
print("✅ Saved predictions to predictions_transformer.csv")
