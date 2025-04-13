import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU, ELU, Concatenate
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import AdamW

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

# Load data
train_df = pd.read_csv('trainCleaned.csv')
train_df, num_scaler, cat_encoders, target_encoder = preprocess_df(train_df, is_train=True)

X = train_df.drop(columns=['Credit_Score'])
y = train_df['Credit_Score']
num_classes = len(np.unique(y))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_cat = to_categorical(y_train, num_classes=num_classes)
y_val_cat = to_categorical(y_val, num_classes=num_classes)

# Manual class weights (tuned for minority focus)
class_weight_dict = {0: 1.0, 1: 2.5, 2: 3.5}

# Input Layer
inp = Input(shape=(X_train.shape[1],))

# Branch A (ELU Path)
x1 = Dense(512)(inp)
x1 = BatchNormalization()(x1)
x1 = ELU()(x1)
x1 = Dropout(0.4)(x1)

# Branch B (LeakyReLU Path)
x2 = Dense(512)(inp)
x2 = BatchNormalization()(x2)
x2 = LeakyReLU(alpha=0.01)(x2)
x2 = Dropout(0.4)(x2)

# Concatenate skip connection
concat = Concatenate()([x1, x2])

# Deep layers
x = Dense(256)(concat)
x = BatchNormalization()(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dropout(0.3)(x)

x = Dense(128)(x)
x = ELU()(x)
x = Dropout(0.2)(x)

x = Dense(64)(x)
x = LeakyReLU(alpha=0.01)(x)
x = Dropout(0.2)(x)

out = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=inp, outputs=out)

model.compile(optimizer=AdamW(learning_rate=0.001, weight_decay=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5)
early_stop = EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)

# Train
history = model.fit(X_train, y_train_cat,
                    validation_data=(X_val, y_val_cat),
                    epochs=100,
                    batch_size=32,
                    class_weight=class_weight_dict,
                    callbacks=[lr_scheduler, early_stop],
                    verbose=1)

# Save training history
pd.DataFrame(history.history).to_csv('training_mlp.csv', index=False)
print("✅ Training history saved to training_mlp.csv")

# Predict on test
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
print("✅ Predictions saved to predictions_mlp.csv")

