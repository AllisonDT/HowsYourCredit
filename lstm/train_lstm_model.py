
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU (safe fallback)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load data
df = pd.read_csv('trainEnhanced.csv')
df.sort_values(by=['Customer_ID', 'Month'], inplace=True)

# Sequence creation
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

X_sequences = np.array(X_sequences)
y_labels = np.array(y_labels)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_labels)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(
    X_sequences, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Build model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile
optimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
early_stop = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate
val_loss, val_accuracy = model.evaluate(X_val, y_val)

# Predict and evaluate
y_pred_probs = model.predict(X_val)
y_pred = np.argmax(y_pred_probs, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, output_dict=True)

# Manually extract binary metrics from confusion matrix if it's binary
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Confusion Matrix: TP={tp}, FN={fn}, FP={fp}, TN={tn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
else:
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("Classification Report:")
    print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
