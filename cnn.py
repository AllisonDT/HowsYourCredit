import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ---------------------------
# 1. Load and Preprocess Data
# ---------------------------
def preprocess_df(df, is_train=True, num_scaler=None, cat_encoders=None, target_encoder=None):
    # Drop columns that are identifiers or less useful for prediction
    drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Define columns by type
    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary',
                    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                    'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
                    'Credit_Utilization_Ratio', 'Credit_History_Age',
                    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']
    
    categorical_cols = ['Type_of_Loan', 'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount']
    
    # Convert numeric columns to numeric and fill missing values with median
    # Convert numeric columns to numeric and fill missing values with median
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
     
    # For categorical columns, fill missing with mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # If training, fit scalers/encoders; else transform using passed objects
    if is_train:
        # Scale numeric features
        num_scaler = StandardScaler()
        df[numeric_cols] = num_scaler.fit_transform(df[numeric_cols])
        
        # Encode categorical features
        cat_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            cat_encoders[col] = le
        
        # If target is present, encode it
        if 'Credit_Score' in df.columns:
            target_encoder = LabelEncoder()
            df['Credit_Score'] = target_encoder.fit_transform(df['Credit_Score'])
    else:
        # For test data: use the provided scaler and encoders
        if num_scaler is not None:
            df[numeric_cols] = num_scaler.transform(df[numeric_cols])
        if cat_encoders is not None:
            for col in categorical_cols:
                df[col] = cat_encoders[col].transform(df[col])
                
    return df, num_scaler, cat_encoders, target_encoder

# Load training data (adjust file path as needed)
train_df = pd.read_csv('trainCleaned.csv')
train_df, num_scaler, cat_encoders, target_encoder = preprocess_df(train_df, is_train=True)

# Separate features and target
if 'Credit_Score' not in train_df.columns:
    raise ValueError("Training data must include the Credit_Score column as target.")
X = train_df.drop(columns=['Credit_Score'])
y = train_df['Credit_Score']

# Optionally, split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to numpy arrays and then reshape for CNN input.
# CNNs expect a 3D input: (samples, timesteps, channels)
# Here, treat each feature as a “time step” and use 1 channel.
X_train_np = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_np   = X_val.to_numpy().reshape((X_val.shape[0], X_val.shape[1], 1))

# Convert target to categorical (one-hot encoding)
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# ---------------------------
# 2. Build the CNN Model
# ---------------------------
model = Sequential([
    Input(shape=(X_train_np.shape[1], 1)),
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# 3. Train the Model
# ---------------------------
history = model.fit(X_train_np, y_train_cat, 
                    validation_data=(X_val_np, y_val_cat),
                    epochs=200, batch_size=8, verbose=1)

# ---------------------------
# 4. Load and Preprocess Testing Data
# ---------------------------
# Load testing data (adjust file path as needed)
test_df = pd.read_csv('testCleaned.csv')
# Keep a copy of IDs to include in the output
id_df = test_df[['ID']].copy()

# Preprocess test data using the scalers/encoders fitted on training data
test_df, _, _, _ = preprocess_df(test_df, is_train=False, 
                                   num_scaler=num_scaler, 
                                   cat_encoders=cat_encoders)

# Make sure the test_df has the same feature columns as X
X_test = test_df  # all columns in test_df are used as features
X_test_np = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

# ---------------------------
# 5. Make Predictions and Save to CSV
# ---------------------------
# Predict probabilities then pick the class with highest probability
pred_probs = model.predict(X_test_np)
pred_class_indices = np.argmax(pred_probs, axis=1)
# Convert numeric labels back to original credit rating names
pred_labels = target_encoder.inverse_transform(pred_class_indices)

# Create a DataFrame for output with ID and predicted Credit Score
output_df = id_df.copy()
output_df['Predicted_Credit_Score'] = pred_labels

# Save predictions to CSV
output_df.to_csv('predictions.csv', index=False)

print("Predictions saved to predictions.csv")
