
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_csv('testCleaned.csv')

# Drop unnecessary columns
df.drop(columns=['ID', 'SSN', 'Name', 'Monthly_Balance'], inplace=True)

# Fill missing values
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

# Encode categorical columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Add cyclical month features
month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12
}
df['Month_Num'] = df['Month'].map(month_map)
df['Month_sin'] = np.sin(2 * np.pi * df['Month_Num'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month_Num'] / 12)

# Add delta features
df['Debt_Change'] = df.groupby('Customer_ID')['Outstanding_Debt'].diff().fillna(0)
df['Credit_Limit_Change'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].diff().fillna(0)

# Add ratio features
df['Debt_to_Income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-5)
df['Salary_to_Installments'] = df['Monthly_Inhand_Salary'] / (df['Total_EMI_per_month'] + 1e-5)

# Add rolling features
df['Outstanding_Debt_rolling_mean'] = df.groupby('Customer_ID')['Outstanding_Debt'].transform(lambda x: x.rolling(3).mean()).bfill()
df['Credit_Utilization_Ratio_rolling_mean'] = df.groupby('Customer_ID')['Credit_Utilization_Ratio'].transform(lambda x: x.rolling(3).mean()).bfill()
df['Num_Credit_Inquiries_rolling_std'] = df.groupby('Customer_ID')['Num_Credit_Inquiries'].transform(lambda x: x.rolling(3).std()).fillna(0)
df['Changed_Credit_Limit_rolling_std'] = df.groupby('Customer_ID')['Changed_Credit_Limit'].transform(lambda x: x.rolling(3).std()).fillna(0)

# Normalize features
numeric_features = df.select_dtypes(include=['float64', 'int64']).columns
final_numeric_cols = numeric_features.drop('Credit_Score') if 'Credit_Score' in numeric_features else numeric_features
scaler = MinMaxScaler()
df[final_numeric_cols] = scaler.fit_transform(df[final_numeric_cols])

# Drop intermediate columns and save final dataset
df.drop(columns=['Month_Num'], inplace=True)
df.to_csv('testEnhanced.csv', index=False)

print("Enhanced dataset saved as 'testEnhanced.csv'")
