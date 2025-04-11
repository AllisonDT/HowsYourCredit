import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv1D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ---------------------------
# 1. Load and Preprocess Data with Enhanced Feature Engineering & Cleaning
# ---------------------------
def preprocess_df(df, is_train=True, num_scaler=None, cat_encoders=None, target_encoder=None):
    # Drop identifier or less useful columns
    drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # ---------------------------
    # Feature Engineering
    # ---------------------------
    df['DebtToIncome'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-5)
    df['Total_Bank_Products'] = df['Num_Bank_Accounts'] + df['Num_Credit_Card']
    df['CreditInquiry_Ratio'] = df['Num_Credit_Inquiries'] / (df['Num_of_Loan'] + 1)
    df['Delayed_Payment_Ratio'] = df['Num_of_Delayed_Payment'] / (df['Delay_from_due_date'] + 1)
    df['CreditHistory_Age_Ratio'] = df['Credit_History_Age'] / (df['Age'] + 1e-5)
    df['Income_to_EMI_Ratio'] = df['Monthly_Inhand_Salary'] / (df['Total_EMI_per_month'] + 1e-5)
    
    # Replace any infinity values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ---------------------------
    # Define columns by type
    # ---------------------------
    numeric_cols = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary',
        'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
        'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
        'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
        'Credit_Utilization_Ratio', 'Credit_History_Age',
        'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance',
        'DebtToIncome', 'Total_Bank_Products', 'CreditInquiry_Ratio',
        'Delayed_Payment_Ratio', 'CreditHistory_Age_Ratio', 'Income_to_EMI_Ratio'
    ]
    
    categorical_cols = ['Type_of_Loan', 'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount']
    
    # ---------------------------
    # Convert numeric columns and fill missing values with the median
    # ---------------------------
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
     
    # For categorical columns, fill missing values with the mode
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    # ---------------------------
    # Fit or transform scalers and encoders based on is_train flag
    # ---------------------------
    if is_train:
        num_scaler = StandardScaler()
        df[numeric_cols] = num_scaler.fit_transform(df[numeric_cols])
        
        cat_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            cat_encoders[col] = le
        
        if 'Credit_Score' in df.columns:
            target_encoder = LabelEncoder()
            df['Credit_Score'] = target_encoder.fit_transform(df['Credit_Score'])
    else:
        if num_scaler is not None:
            df[numeric_cols] = num_scaler.transform(df[numeric_cols])
        if cat_encoders is not None:
            for col in categorical_cols:
                df[col] = cat_encoders[col].transform(df[col])
                
    return df, num_scaler, cat_encoders, target_encoder

# Load and preprocess training data
train_df = pd.read_csv('trainCleaned.csv')
train_df, num_scaler, cat_encoders, target_encoder = preprocess_df(train_df, is_train=True)

if 'Credit_Score' not in train_df.columns:
    raise ValueError("Training data must include the Credit_Score column as target.")

# Separate features and target
X = train_df.drop(columns=['Credit_Score'])
y = train_df['Credit_Score']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to numpy arrays and reshape for CNN input (samples, timesteps, channels)
X_train_np = X_train.to_numpy().reshape((X_train.shape[0], X_train.shape[1], 1))
X_val_np   = X_val.to_numpy().reshape((X_val.shape[0], X_val.shape[1], 1))

# One-hot encode the targets
num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# ---------------------------
# Custom Learnable Feature Weighting Layer Definition
# ---------------------------
class FeatureWeightingLayer(tf.keras.layers.Layer):
    def __init__(self, num_features, **kwargs):
        super(FeatureWeightingLayer, self).__init__(**kwargs)
        self.num_features = num_features

    def build(self, input_shape):
        self.feature_weights = self.add_weight(
            shape=(self.num_features,), 
            initializer='ones', 
            trainable=True, 
            name='feature_weights'
        )
        super(FeatureWeightingLayer, self).build(input_shape)

    def call(self, inputs):
        squeezed = tf.squeeze(inputs, axis=-1)
        weighted = squeezed * self.feature_weights
        return tf.expand_dims(weighted, axis=-1)

# ---------------------------
# Define Custom Callback to Log Epoch Metrics
# ---------------------------
class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super(MetricsLogger, self).__init__()
        self.validation_data = validation_data  # (X_val_np, y_val_cat)
        self.logs_list = []
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Get the validation data and corresponding true labels
        val_X, val_y_cat = self.validation_data
        val_y = np.argmax(val_y_cat, axis=1)
        
        # Obtain model predictions on the validation set
        pred_probs = self.model.predict(val_X, verbose=0)
        pred_labels = np.argmax(pred_probs, axis=1)
        
        # Compute the confusion matrix
        cm = confusion_matrix(val_y, pred_labels)
        if cm.shape == (2, 2):  # Binary classification
            tn, fp, fn, tp = cm.ravel()
        else:
            total = cm.sum()
            TP = np.diag(cm)
            FP = cm.sum(axis=0) - TP
            FN = cm.sum(axis=1) - TP
            TN = total - (TP + FP + FN)
            tp = np.mean(TP)
            fp = np.mean(FP)
            fn = np.mean(FN)
            tn = np.mean(TN)
        
        # Compute precision, recall, and F1 score
        if cm.shape == (2, 2):
            precision = precision_score(val_y, pred_labels, average='binary')
            recall = recall_score(val_y, pred_labels, average='binary')
            f1 = f1_score(val_y, pred_labels, average='binary')
        else:
            precision = precision_score(val_y, pred_labels, average='macro')
            recall = recall_score(val_y, pred_labels, average='macro')
            f1 = f1_score(val_y, pred_labels, average='macro')
        
        # Log the metrics for the epoch
        entry = {
            'epoch': epoch + 1,
            'val_accuracy': logs.get('val_accuracy'),
            'val_loss': logs.get('val_loss'),
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'tn': tn,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        self.logs_list.append(entry)
        
    def on_train_end(self, logs=None):
        df_metrics = pd.DataFrame(self.logs_list)
        df_metrics.to_csv('epoch_metrics_cnn.csv', index=False)
        print('Epoch metrics saved to epoch_metrics_cnn.csv')

# ---------------------------
# 2. Build the Enhanced CNN Model with Feature Weighting, Batch Normalization, and Adjusted Dropout
# ---------------------------
model = Sequential([
    Input(shape=(X_train_np.shape[1], 1)),
    FeatureWeightingLayer(X_train_np.shape[1]),
    
    # First convolutional block
    Conv1D(filters=128, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    # Second convolutional block
    Conv1D(filters=64, kernel_size=2, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Flatten(),
    
    # Fully connected block
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(num_classes, activation='softmax')
])

# Compile the model with adam optimizer and categorical crossentropy
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# ---------------------------
# 3. Train the Enhanced Model with Early Stopping, Learning Rate Scheduler, and Metrics Logging
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
metrics_logger = MetricsLogger(validation_data=(X_val_np, y_val_cat))

history = model.fit(
    X_train_np, y_train_cat, 
    validation_data=(X_val_np, y_val_cat),
    epochs=100,         # You can adjust the max epochs if required.
    batch_size=16,     # Experiment with batch size if needed.
    verbose=1,
    callbacks=[early_stop, lr_scheduler, metrics_logger]
)

# ---------------------------
# Save Training Visualizations to PNG Files
# ---------------------------

# 1. Plot Loss vs. Epoch
# plt.figure()
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss vs. Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig('loss_vs_epoch.png')
# plt.close()

# # 2. Plot Accuracy vs. Epoch
# plt.figure()
# plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
# plt.title('Accuracy vs. Epoch')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.savefig('accuracy_vs_epoch.png')
# plt.close()

# ... (your training and visualization code above)

# 3. Plot Confusion Matrix on the Validation Set
val_preds = np.argmax(model.predict(X_val_np, verbose=0), axis=1)
val_true = np.argmax(y_val_cat, axis=1)
cm = confusion_matrix(val_true, val_preds)
plt.figure()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predicted label')
plt.ylabel('True label')
# Add cell text
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max()/2. else "black")
plt.savefig('confusion_matrix_cnn.png')
plt.close()

# Convert the logged metrics from the custom callback into a DataFrame
metrics_df = pd.DataFrame(metrics_logger.logs_list)
print(metrics_df)  # Optional: Check the DataFrame in console

plt.figure(figsize=(10, 6))
plt.plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Validation Accuracy')
plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
plt.plot(metrics_df['epoch'], metrics_df['precision'], label='Precision')
plt.plot(metrics_df['epoch'], metrics_df['recall'], label='Recall')
plt.plot(metrics_df['epoch'], metrics_df['f1'], label='F1 Score')
plt.title('Epoch vs. Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.legend()
plt.savefig('epoch_vs_metrics_cnn.png')
plt.close()

# ---------------------------
# 4. Load and Preprocess Testing Data Using the Same Pipeline
# ---------------------------
test_df = pd.read_csv('testCleaned.csv')
id_df = test_df[['ID']].copy()  # Save the IDs for final output

test_df, _, _, _ = preprocess_df(test_df, is_train=False, 
                                   num_scaler=num_scaler, 
                                   cat_encoders=cat_encoders)

X_test = test_df  # Use all columns in test_df as features
X_test_np = X_test.to_numpy().reshape((X_test.shape[0], X_test.shape[1], 1))

# ---------------------------
# 5. Make Predictions and Save to CSV
# ---------------------------
pred_probs = model.predict(X_test_np)
pred_class_indices = np.argmax(pred_probs, axis=1)
pred_labels = target_encoder.inverse_transform(pred_class_indices)

output_df = id_df.copy()
output_df['Predicted_Credit_Score'] = pred_labels
output_df.to_csv('predictions_cnn.csv', index=False)
print("Predictions saved to predictions_cnn.csv")
