import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------------------
# 1. Load Your Data
# ---------------------------
# Replace 'train.csv' and 'test.csv' with your actual file paths.
train_data = pd.read_csv('trainCleaned.csv')
test_data = pd.read_csv('testCleaned.csv')

# ---------------------------
# 2. Preprocess the Data
# ---------------------------
# For training data, separate the features (X) and the target (y)
# We assume 'Credit_Score' is the target and 'ID' is an identifier.
# Drop columns that are not used as features.
X = train_data.drop(columns=['Credit_Score', 'ID'])
y = train_data['Credit_Score']

# For testing data, drop the identifier column if needed.
X_test = test_data.drop(columns=['ID'], errors='ignore')

# (Optional) Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# ---------------------------
# 3. Reshape for CNN Input
# ---------------------------
# CNN layers expect a 3D input: (samples, time_steps, features).
# Here we treat each feature as a "time step" by reshaping the data.
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# ---------------------------
# 4. Build the CNN Model
# ---------------------------
model = Sequential([
    # 1D Convolution: adjust filters and kernel_size as needed
    Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=(X_cnn.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1)  # Regression output (credit score)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# ---------------------------
# 5. Train the Model
# ---------------------------
# Split training data to monitor validation performance
X_train, X_val, y_train, y_val = train_test_split(X_cnn, y, test_size=0.2, random_state=42)

# Early stopping to avoid overfitting
es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

model.fit(X_train, y_train,
          epochs=100,
          batch_size=32,
          validation_data=(X_val, y_val),
          callbacks=[es])

# ---------------------------
# 6. Predict on Test Data
# ---------------------------
predictions = model.predict(X_test_cnn)

# ---------------------------
# 7. Save Predictions to CSV
# ---------------------------
# Create a DataFrame with columns "id" and "credit score"
output_df = pd.DataFrame({
    'id': test_data['ID'],
    'credit score': predictions.flatten()
})

# Save to CSV
output_df.to_csv('predicted_credit_scores_cnn.csv', index=False)
print("Predictions saved to predicted_credit_scores.csv")
