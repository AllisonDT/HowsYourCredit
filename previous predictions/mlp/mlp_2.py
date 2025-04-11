import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

def preprocess_df(df, is_train=True, num_scaler=None, cat_encoders=None, target_encoder=None):
    drop_cols = ['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Occupation']
    df = df.drop(columns=drop_cols, errors='ignore')

    numeric_cols = ['Age', 'Annual_Income', 'Monthly_Inhand_Salary',
                    'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate',
                    'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment',
                    'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt',
                    'Credit_Utilization_Ratio', 'Credit_History_Age',
                    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

    categorical_cols = ['Type_of_Loan', 'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount']

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

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

X = train_df.drop(columns=['Credit_Score'])
y = train_df['Credit_Score']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(np.unique(y))
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# Class weights to balance training
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)
class_weight_dict = dict(enumerate(class_weights))

# Build MLP model with LeakyReLU
model = Sequential()
model.add(Dense(256, input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.4))

model.add(Dense(128))
model.add(BatchNormalization())
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.3))

model.add(Dense(64))
model.add(LeakyReLU(alpha=0.01))
model.add(Dropout(0.3))

model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks: reduce LR and stop early if val_loss doesn't improve
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train model
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=50,
                    batch_size=16,
                    class_weight=class_weight_dict,
                    callbacks=[lr_scheduler, early_stop],
                    verbose=1)

# Save training history
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_mlp.csv', index=False)
print("Training history saved to training_mlp.csv")

# Predict on test data
test_df = pd.read_csv('testCleaned.csv')
id_df = test_df[['ID']].copy()

test_df, _, _, _ = preprocess_df(test_df, is_train=False, 
                                 num_scaler=num_scaler, 
                                 cat_encoders=cat_encoders)

X_test = test_df
pred_probs = model.predict(X_test)
pred_class_indices = np.argmax(pred_probs, axis=1)
pred_labels = target_encoder.inverse_transform(pred_class_indices)

output_df = id_df.copy()
output_df['Predicted_Credit_Score'] = pred_labels
output_df.to_csv('predictions_mlp.csv', index=False)
print("Predictions saved to predictions_mlp.csv")
