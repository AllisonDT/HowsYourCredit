import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --------------------------
# 1. Load Data
# --------------------------
df = pd.read_csv("trainCleaned.csv")

# --------------------------
# 2. Drop Garbage Columns
# --------------------------
df = df.drop(columns=[
    'ID', 'Customer_ID', 'Name', 'SSN', 'Type_of_Loan', 'Monthly_Balance'
], errors='ignore')

# --------------------------
# 3. Filter Obvious Outliers
# --------------------------
df = df[(df['Age'] > 0) & (df['Age'] < 120)]
df = df[(df['Interest_Rate'] >= 0) & (df['Interest_Rate'] < 100)]
df = df[(df['Num_Bank_Accounts'] >= 0) & (df['Num_Bank_Accounts'] < 100)]
df = df[(df['Num_of_Loan'] >= 0) & (df['Num_of_Loan'] < 50)]

# --------------------------
# 4. Separate Labels
# --------------------------
y = df['Credit_Score']
X = df.drop(columns=['Credit_Score'])

# --------------------------
# 5. Identify Columns
# --------------------------
categorical_cols = X.select_dtypes(include='object').columns.tolist()
numeric_cols = X.select_dtypes(include='number').columns.tolist()

# --------------------------
# 6. Encode Categorical Features
# --------------------------
for col in categorical_cols:
    X[col] = X[col].astype('category').cat.codes



# --------------------------
# 7. Scale Numerical Features
# --------------------------
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# --------------------------
# 8. Final Cleanup
# --------------------------
X = X.dropna()
y = y[X.index]

X = X.astype(np.float32)
y = y.astype('category').cat.codes.astype(np.int64)  # turn labels into 0/1/2

# --------------------------
# 9. Train/Validation Split
# --------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# 10. Save as .npy
# --------------------------
np.save("X_train.npy", X_train.to_numpy())
np.save("X_val.npy", X_val.to_numpy())
np.save("y_train.npy", y_train.to_numpy())
np.save("y_val.npy", y_val.to_numpy())

print("✅ Preprocessing complete. Files saved:")
print("- X_train.npy")
print("- X_val.npy")
print("- y_train.npy")
print("- y_val.npy")
