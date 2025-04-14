import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import PowerTransformer


# Load clean files
train_df = pd.read_csv('trainCleaned.csv')
test_df = pd.read_csv('testCleaned.csv')

# Copy for personal experimentation
my_train = train_df.copy()
my_test = test_df.copy()

# Feature engineering
for df in [my_train, my_test]:
    # Ensure numerics are numeric
    for col in [
        'Outstanding_Debt', 'Changed_Credit_Limit', 'Monthly_Balance',
        'Amount_invested_monthly', 'Annual_Income', 'Num_Credit_Card',
        'Interest_Rate', 'Num_Bank_Accounts', 'Age'
    ]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Original features
    df['debt_to_income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + 1e-6)
    df['credit_use_ratio'] = df['Credit_Utilization_Ratio'] * df['Num_Credit_Card']
    df['age_credit_mix'] = df['Age'] * df['Credit_Mix'].apply(lambda x: 0 if pd.isna(x) else ord(str(x)[0]) - ord('A') + 1)
    df['debt_credit_interaction'] = df['debt_to_income'] * df['credit_use_ratio']
    
    # NEW features
    df['util_per_card'] = df['Credit_Utilization_Ratio'] / (df['Num_Credit_Card'] + 1e-6)
    df['avg_investment_ratio'] = df['Amount_invested_monthly'] / (df['Monthly_Balance'] + 1e-6)
    df['loan_count'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) else 0)
    df['credit_limit_change_ratio'] = df['Changed_Credit_Limit'] / (df['Outstanding_Debt'] + 1e-6)
    df['investment_income_ratio'] = df['Amount_invested_monthly'] / (df['Annual_Income'] + 1e-6)
    df['monthly_burden'] = df['Monthly_Balance'] / (df['Num_Bank_Accounts'] + 1e-6)
    df['income_per_age'] = df['Annual_Income'] / (df['Age'] + 1e-6)
    df['log_debt_to_income'] = np.log1p(df['debt_to_income'])
    df['log_util_per_card'] = np.log1p(df['util_per_card'])
    df['risk_factor'] = df['Monthly_Balance'] / (df['Num_Credit_Card'] + df['Num_Bank_Accounts'] + 1e-6)
    df['investment_pressure'] = df['Amount_invested_monthly'] / (df['Monthly_Balance'] + df['Outstanding_Debt'] + 1e-6)


    # Clip outliers
    for col in ['Outstanding_Debt', 'Changed_Credit_Limit', 'Monthly_Balance']:
        df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))

    # Log transform
    for col in ['Outstanding_Debt', 'Monthly_Balance', 'Amount_invested_monthly']:
        df[col] = np.log1p(df[col])

    # Discretize interest rate
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
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

# Apply PowerTransformer to numeric features
pt = PowerTransformer()
X_train_numeric_raw = my_train[numeric_cols].copy()
X_test_numeric_raw = my_test[numeric_cols].copy()

my_train[numeric_cols] = pt.fit_transform(X_train_numeric_raw)
my_test[numeric_cols] = pt.transform(X_test_numeric_raw)


# Encode categoricals
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = ohe.fit_transform(my_train[categorical_cols])
X_test_cat = ohe.transform(my_test[categorical_cols])

# Normalize numeric
scaler = StandardScaler()
X_train_num = scaler.fit_transform(my_train[numeric_cols])
X_test_num = scaler.transform(my_test[numeric_cols])

# Combine features
X_train_full = np.hstack([X_train_num, X_train_cat])
X_test_full = np.hstack([X_test_num, X_test_cat])
y = my_train['Credit_Score'].to_numpy()

# Split
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42)

# Class weights
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Callbacks
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.4, patience=3, min_lr=1e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

# Build model
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(512, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation(swish),
    Dropout(0.3),

    Dense(256, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation(swish),
    Dropout(0.3),

    Dense(128, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation(swish),
    Dropout(0.2),

    Dense(64, kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Activation(swish),
    Dropout(0.2),

    Dense(3, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=32,
                    class_weight=class_weights,
                    callbacks=[lr_schedule, early_stop],
                    verbose=1)

# Evaluate
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_val, y_pred_classes)
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)

precision = precision_score(y_val, y_pred_classes, average='macro')
recall = recall_score(y_val, y_pred_classes, average='macro')
f1 = f1_score(y_val, y_pred_classes, average='macro')

# Save training metrics
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

# Predict test
pred_probs = model.predict(X_test_full)
pred_classes = np.argmax(pred_probs, axis=1)
pred_labels = target_encoder.inverse_transform(pred_classes)

output_df = my_test[['ID']].copy()
output_df['Predicted_Credit_Score'] = pred_labels
output_df.to_csv('predictions_mlp.csv', index=False)
print("✅ predictions_mlp.csv saved")