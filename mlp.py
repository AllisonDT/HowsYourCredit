import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization, ReLU, Input
from tensorflow.keras.optimizers import Adam

# Load shared files (safe)
train_df = pd.read_csv('trainCleaned.csv')
test_df = pd.read_csv('testCleaned.csv')

# Copy for feature engineering (only used in your code)
my_train = train_df.copy()
my_test = test_df.copy()

# Create new features
for df in [my_train, my_test]:
    df['debt_to_income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-6)
    df['credit_use_ratio'] = df['Credit_Utilization_Ratio'] * df['Num_Credit_Card']
    df['age_credit_mix'] = df['Age'] * df['Credit_Mix'].apply(lambda x: 0 if pd.isna(x) else ord(str(x)[0]) - ord('A') + 1)
    for col in ['Outstanding_Debt', 'Changed_Credit_Limit', 'Monthly_Balance']:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # convert to float
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
    for col in ['Outstanding_Debt', 'Monthly_Balance', 'Amount_invested_monthly']:
        df[col] = np.log1p(df[col])
    df['interest_bin'] = pd.cut(df['Interest_Rate'], bins=[-1, 5, 15, 50], labels=['low', 'med', 'high'])

# Encode target
target_encoder = LabelEncoder()
my_train['Credit_Score'] = target_encoder.fit_transform(my_train['Credit_Score'])

# Define columns
categorical_cols = ['Type_of_Loan', 'Payment_Behaviour', 'Credit_Mix', 'Payment_of_Min_Amount', 'interest_bin']
numeric_cols = my_train.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'Credit_Score']

# Fill NA
for df in [my_train, my_test]:
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

# Encode categoricals
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(my_train[categorical_cols])
X_test_cat = ohe.transform(my_test[categorical_cols])

# Normalize numeric
scaler = StandardScaler()
X_train_num = scaler.fit_transform(my_train[numeric_cols])
X_test_num = scaler.transform(my_test[numeric_cols])

# Final features
X_train_full = np.hstack([X_train_num, X_train_cat])
X_test_full = np.hstack([X_test_num, X_test_cat])
y = my_train['Credit_Score']

# Train/val split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42)

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import swish

# Refined MLP with Swish, BN, and new layer sizes
model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))

# Layer 1: Wide + Normalize + Swish
model.add(Dense(256))
model.add(BatchNormalization())
model.add(ReLU())  # swish may also work here, see below
model.add(Dropout(0.3))

# Layer 2: Mid
model.add(Dense(128))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.3))

# Layer 3: Narrow
model.add(Dense(64))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.2))

# Output
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,
                    batch_size=32,
                    verbose=1)

# Predict & evaluate
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(y_val, y_pred_classes)
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)

# Precision, Recall, F1
precision = precision_score(y_val, y_pred_classes, average='macro')
recall = recall_score(y_val, y_pred_classes, average='macro')
f1 = f1_score(y_val, y_pred_classes, average='macro')

# Save metrics
df_metrics = pd.DataFrame({
    'val_accuracy': [history.history['val_accuracy'][-1]],
    'val_loss': [history.history['val_loss'][-1]],
    'cm_tp': [tp.sum()],
    'cm_fn': [fn.sum()],
    'cm_fp': [fp.sum()],
    'cm_tn': [tn.sum()],
    'precision': [precision],
    'recall': [recall],
    'f1_score': [f1]
})
df_metrics.to_csv('training_mlp.csv', index=False)
print("✅ training_mlp.csv saved")

# Predict on test and save
pred_probs = model.predict(X_test_full)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = target_encoder.inverse_transform(pred_classes)

output_df = my_test[['ID']].copy()
output_df['Predicted_Credit_Score'] = pred_labels
output_df.to_csv('predictions_mlp.csv', index=False)
print("✅ predictions_mlp.csv saved")

