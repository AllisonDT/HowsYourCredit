
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ====== TRAINING ======
print("Loading and preparing training data...")
train_df = pd.read_csv('C:\\Users\\JordanJames\\Documents\\pyth\\trainEnhanced.csv')
train_df.sort_values(by=['Customer_ID', 'Month'], inplace=True)

sequence_length = 3
X_train_seq, y_train_seq = [], []
for _, group in train_df.groupby('Customer_ID'):
    if len(group) >= sequence_length:
        for i in range(len(group) - sequence_length + 1):
            seq = group.iloc[i:i+sequence_length]
            X = seq.drop(columns=[col for col in ['Customer_ID', 'Credit_Score', 'Month'] if col in seq.columns]).values
            y = seq['Credit_Score'].iloc[-1]
            X_train_seq.append(X)
            y_train_seq.append(y)

X_train_seq = np.array(X_train_seq)
y_train_seq = [str(label) for label in y_train_seq]
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_seq)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_seq, y_train_encoded, test_size=0.2, random_state=42, stratify=y_train_encoded
)

print("Training model...")
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

model.save_weights("lstm_model.weights.h5")
print("Model training complete. Weights saved to 'lstm_model.weights.h5'.")

# ====== INFERENCE ======
print("Loading and preparing test data...")
test_df = pd.read_csv('C:\\Users\\JordanJames\\Documents\\pyth\\testEnhanced.csv')
original_test = pd.read_csv('C:\\Users\\JordanJames\\Documents\\pyth\\test.csv')
test_df['ID'] = original_test['ID']

test_df.sort_values(by=['Customer_ID', 'Month'], inplace=True)

X_test_seq = []
test_ids = []

for customer_id, group in test_df.groupby('Customer_ID'):
    if len(group) >= sequence_length:
        for i in range(len(group) - sequence_length + 1):
            seq = group.iloc[i:i+sequence_length]
            X = seq.drop(columns=[col for col in ['Customer_ID', 'Credit_Score', 'Month', 'ID'] if col in seq.columns]).values
            X_test_seq.append(X)
            test_ids.append(seq['ID'].iloc[-1])

X_test_seq = np.array(X_test_seq)

print("Loading model weights and running predictions...")
model.load_weights("lstm_model.weights.h5")

y_test_probs = model.predict(X_test_seq)
y_test_pred = np.argmax(y_test_probs, axis=1)
y_test_labels = label_encoder.inverse_transform(y_test_pred)

output_df = pd.DataFrame({
    'ID': test_ids,
    'Predicted_Credit_Score': y_test_labels
}).drop_duplicates(subset='ID', keep='last')

output_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to 'test_predictions.csv'")

# Prepare output
output_df = pd.DataFrame({
    'ID': test_ids,
    'Predicted_Credit_Score': y_test_labels
}).drop_duplicates(subset='ID', keep='last')

output_df.to_csv("test_predictions.csv", index=False)
print("Predictions saved to 'test_predictions.csv'")
