
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd




early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Load enhanced training data
df = pd.read_csv('trainEnhanced.csv')

# Sort and prepare sequences
df.sort_values(by=['Customer_ID', 'Month'], inplace=True)

# Group by customer and generate sequences
sequence_length = 5
X_sequences, y_labels = [], []

for _, group in df.groupby('Customer_ID'):
    if len(group) >= sequence_length:
        for i in range(len(group) - sequence_length + 1):
            seq = group.iloc[i:i+sequence_length]
            X = seq.drop(columns=['Customer_ID', 'Credit_Score', 'Month']).values
            y = seq['Credit_Score'].iloc[-1]
            X_sequences.append(X)
            y_labels.append(y)

# Convert to numpy arrays
X_sequences = np.array(X_sequences)
y_labels = np.array(y_labels)

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    # callbacks=[early_stop]
)

# Evaluate
loss, accuracy = model.evaluate(X_val, y_val)
pd.DataFrame(history.history).to_csv("training_log.csv", index=False)
print(f"Validation Accuracy: {accuracy:.4f}")
