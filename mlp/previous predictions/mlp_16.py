import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold # Added StratifiedKFold for robustness check suggestion
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score # Added accuracy_score
from sklearn.utils import class_weight
import tensorflow as tf # Use explicit tf import
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation, Input
from tensorflow.keras.regularizers import l2 # Kept for potential comparison with AdamW
# from tensorflow.keras.optimizers import Adam # Replaced by AdamW
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.activations import swish, gelu # Added gelu as an alternative to try
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load clean files
try:
    train_df = pd.read_csv('trainCleaned.csv')
    test_df = pd.read_csv('testCleaned.csv')
except FileNotFoundError:
    print("Error: trainCleaned.csv or testCleaned.csv not found. Make sure they are in the correct directory.")
    exit()


# Copy for personal experimentation
my_train = train_df.copy()
my_test = test_df.copy()

# --- Feature Engineering ---
print("Starting feature engineering...")
# Define numeric columns (include potentially new ones if they exist)
numeric_base_cols = [
    'Outstanding_Debt', 'Changed_Credit_Limit', 'Monthly_Balance',
    'Amount_invested_monthly', 'Annual_Income', 'Num_Credit_Card',
    'Interest_Rate', 'Num_Bank_Accounts', 'Age', 'Num_of_Loan',
    'Num_Credit_Inquiries', 'Credit_Utilization_Ratio',
    'Total_EMI_per_month', 'Delay_from_due_date'
    # Add 'Monthly_Inhand_Salary' IF IT EXISTS in your CSV
    # 'Monthly_Inhand_Salary'
]
# Check if 'Monthly_Inhand_Salary' exists and add it
if 'Monthly_Inhand_Salary' in my_train.columns:
    numeric_base_cols.append('Monthly_Inhand_Salary')
    print("Included 'Monthly_Inhand_Salary' in numeric processing.")
else:
    print("Warning: 'Monthly_Inhand_Salary' not found in train_df, skipping related features.")


for df in [my_train, my_test]:
    # Ensure numerics are numeric (handle potential errors)
    for col in numeric_base_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
             print(f"Warning: Column '{col}' not found in dataframe, skipping.")

    # Fill NAs BEFORE creating interaction features if components might be NA
    # Let's fill numeric NAs with median here temporarily for feature creation robustness
    # We will properly fill later before scaling
    temp_numeric_cols_in_df = [col for col in numeric_base_cols if col in df.columns]
    for col in temp_numeric_cols_in_df:
         df[col] = df[col].fillna(df[col].median())

    # --- Original & New Features ---
    # (Using small epsilon to prevent division by zero)
    epsilon = 1e-7

    # Income / Debt related
    if 'Annual_Income' in df.columns:
        df['debt_to_income'] = df['Outstanding_Debt'] / (df['Annual_Income'] + epsilon)
        df['investment_income_ratio'] = df['Amount_invested_monthly'] / (df['Annual_Income'] + epsilon)
        df['income_per_age'] = df['Annual_Income'] / (df['Age'] + epsilon)
        if 'Monthly_Inhand_Salary' in df.columns:
             df['debt_to_monthly_income'] = df['Outstanding_Debt'] / (df['Monthly_Inhand_Salary'] * 12 + epsilon) # Approx annual
             df['monthly_inv_to_monthly_income'] = df['Amount_invested_monthly'] / (df['Monthly_Inhand_Salary'] + epsilon)


    # Credit Usage / Limit related
    df['credit_use_ratio_per_card'] = df['Credit_Utilization_Ratio'] / (df['Num_Credit_Card'] + epsilon)
    df['debt_per_card'] = df['Outstanding_Debt'] / (df['Num_Credit_Card'] + epsilon)
    df['balance_to_limit_change'] = df['Monthly_Balance'] / (df['Changed_Credit_Limit'] + epsilon) # Ratio might be tricky if limit change is 0 or negative

    # Balance / Investment related
    df['avg_investment_ratio'] = df['Amount_invested_monthly'] / (df['Monthly_Balance'] + epsilon)
    df['monthly_balance_per_account'] = df['Monthly_Balance'] / (df['Num_Bank_Accounts'] + epsilon)

    # Loan related
    df['loan_count'] = df['Type_of_Loan'].apply(lambda x: len(str(x).split(',')) if pd.notna(x) and str(x).strip() != '' else 0)
    df['balance_per_loan'] = df['Monthly_Balance'] / (df['loan_count'] + epsilon) # Use engineered loan_count

    # Age / Mix related - IMPROVED MAPPING
    # *** IMPORTANT: Adjust this mapping based on your actual 'Credit_Mix' values ***
    credit_mix_mapping = {'Poor': 1, 'Standard': 2, 'Good': 3}
    # Handle potential NaN or unexpected values gracefully
    df['credit_mix_encoded'] = df['Credit_Mix'].map(credit_mix_mapping).fillna(credit_mix_mapping['Standard']) # Fill NA with Standard
    df['age_credit_mix'] = df['Age'] * df['credit_mix_encoded']

    # Interaction
    if 'debt_to_income' in df.columns:
      df['debt_credit_interaction'] = df['debt_to_income'] * df['credit_use_ratio_per_card']

    # --- Transformations & Binning ---

    # Clip outliers (consider adjusting quantiles based on data exploration)
    cols_to_clip = ['Outstanding_Debt', 'Changed_Credit_Limit', 'Monthly_Balance', 'Annual_Income', 'Amount_invested_monthly']
    if 'Monthly_Inhand_Salary' in df.columns:
        cols_to_clip.append('Monthly_Inhand_Salary')

    for col in cols_to_clip:
         if col in df.columns:
            lower_bound = df[col].quantile(0.01)
            upper_bound = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    # Log transform (add 1 to handle zeros) - apply to potentially skewed features
    cols_to_log = ['Outstanding_Debt', 'Monthly_Balance', 'Amount_invested_monthly', 'Annual_Income']
    if 'Monthly_Inhand_Salary' in df.columns:
        cols_to_log.append('Monthly_Inhand_Salary')

    for col in cols_to_log:
        if col in df.columns:
            # Ensure column is numeric and non-negative before log transform
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            df[col] = np.log1p(df[col].apply(lambda x: max(x, 0))) # Ensure non-negative

    # Discretize interest rate (Consider if keeping numeric is better)
    df['interest_bin'] = pd.cut(df['Interest_Rate'], bins=[-1, 10, 20, 35], labels=['low', 'medium', 'high']) # Adjusted bins slightly

print("Feature engineering finished.")

# --- Target Encoding ---
target_encoder = LabelEncoder()
my_train['Credit_Score'] = target_encoder.fit_transform(my_train['Credit_Score'])
print("Target variable encoded.")

# --- Define Columns for Processing ---
# Select all numeric columns INCLUDING newly engineered ones, excluding target
numeric_cols = my_train.select_dtypes(include=np.number).columns.tolist()
numeric_cols = [col for col in numeric_cols if col != 'Credit_Score' and col in my_test.columns] # Ensure cols exist in test too

# Define categorical columns (include engineered bins/encoded)
categorical_cols = ['Type_of_Loan', 'Payment_Behaviour', 'Payment_of_Min_Amount', 'interest_bin', 'credit_mix_encoded'] # Use encoded mix
# Remove original 'Credit_Mix' if it exists and is not needed
if 'Credit_Mix' in numeric_cols: numeric_cols.remove('Credit_Mix')
if 'Credit_Mix' in categorical_cols: categorical_cols.remove('Credit_Mix')

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")


# --- Missing Value Imputation (Final) ---
print("Imputing missing values...")
for df in [my_train, my_test]:
    for col in numeric_cols:
        # Check if column exists before filling NA
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
        else:
             print(f"Warning: Numeric column '{col}' not found for NA fill in {'train' if df is my_train else 'test'} df.")

    for col in categorical_cols:
        if col in df.columns:
            # Use mode().iloc[0] to handle potential multi-modal cases
             mode_val = df[col].mode()
             if not mode_val.empty:
                 df[col] = df[col].fillna(mode_val.iloc[0])
             else:
                 # Handle case where column is all NaN or mode calculation fails
                 df[col] = df[col].fillna('Unknown') # Or another placeholder
                 print(f"Warning: Could not find mode for categorical column '{col}'. Filled NA with 'Unknown'.")
        else:
             print(f"Warning: Categorical column '{col}' not found for NA fill in {'train' if df is my_train else 'test'} df.")

# --- Encoding & Scaling ---
print("Encoding categorical features...")
# Ensure all categorical columns are treated as strings before OHE
for col in categorical_cols:
    if col in my_train.columns:
        my_train[col] = my_train[col].astype(str)
    if col in my_test.columns:
        my_test[col] = my_test[col].astype(str)

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# Fit on training data union test data categories to handle potential unseen values better? Or just fit train. Sticking to train fit.
X_train_cat = ohe.fit_transform(my_train[categorical_cols])
X_test_cat = ohe.transform(my_test[categorical_cols])
ohe_feature_names = ohe.get_feature_names_out(categorical_cols)

print("Scaling numeric features...")
scaler = StandardScaler()
X_train_num = scaler.fit_transform(my_train[numeric_cols])
X_test_num = scaler.transform(my_test[numeric_cols])

# --- Combine Features ---
X_train_full = np.hstack([X_train_num, X_train_cat])
X_test_full = np.hstack([X_test_num, X_test_cat])
y = my_train['Credit_Score'].to_numpy()

feature_names = numeric_cols + list(ohe_feature_names)
print(f"Total features after processing: {X_train_full.shape[1]}")

# --- Train/Validation Split ---
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y, test_size=0.2, random_state=42, stratify=y) # Use stratify for classification

# --- Class Weights ---
classes = np.unique(y_train)
weights = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print(f"Calculated class weights: {class_weights}")

# --- Callbacks ---
# Increased patience slightly
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=5e-6, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1) # Increased patience

# --- Build Model (Adjusted Architecture) ---
def build_classifier(input_shape):
    model = Sequential([
        Input(shape=(input_shape,)),
        Dense(384), # Increased neurons
        BatchNormalization(),
        Activation(gelu), # Try gelu or keep swish
        Dropout(0.4), # Adjusted dropout

        Dense(192), # Increased neurons
        BatchNormalization(),
        Activation(gelu), # Try gelu or keep swish
        Dropout(0.3), # Adjusted dropout

        Dense(96), # Increased neurons
        BatchNormalization(),
        Activation(gelu), # Try gelu or keep swish
        Dropout(0.2),

        Dense(len(classes), activation='softmax') # Output layer size based on number of classes
    ])
    return model

model = build_classifier(X_train.shape[1])

# Using AdamW optimizer
# Note: L2 regularization is handled differently by AdamW (weight decay)
# If using AdamW, typically you don't add kernel_regularizer=l2(...) in Dense layers
optimizer = AdamW(learning_rate=0.0005, weight_decay=0.01) # Adjusted LR, added weight decay

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- Train Model ---
print("Starting model training...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=150, # Increased epochs, relying on EarlyStopping
                    batch_size=64, # Adjusted batch size
                    class_weight=class_weights,
                    callbacks=[lr_schedule, early_stop],
                    verbose=1)

# --- Evaluate ---
print("Evaluating model on validation set...")
# Use the best weights restored by EarlyStopping
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f} ({(val_accuracy * 100):.2f}%)") # Display accuracy

y_pred_probs = model.predict(X_val)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_val, y_pred_classes)
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)

precision = precision_score(y_val, y_pred_classes, average='macro', zero_division=0)
recall = recall_score(y_val, y_pred_classes, average='macro', zero_division=0)
f1 = f1_score(y_val, y_pred_classes, average='macro', zero_division=0)

print(f"Validation Macro Precision: {precision:.4f}")
print(f"Validation Macro Recall: {recall:.4f}")
print(f"Validation Macro F1-Score: {f1:.4f}")
print("Validation Confusion Matrix:\n", cm)


# --- Save Metrics ---
df_metrics = pd.DataFrame({
    'val_accuracy': [val_accuracy], # Use evaluated accuracy
    'val_loss': [val_loss],         # Use evaluated loss
    'cm_tp_sum': [tp.sum()],        # Summing TP across classes
    'cm_fn_sum': [fn.sum()],        # Summing FN across classes
    'cm_fp_sum': [fp.sum()],        # Summing FP across classes
    'cm_tn_sum': [tn.sum()],        # Summing TN across classes (be careful with multi-class TN interpretation)
    'precision_macro': [precision],
    'recall_macro': [recall],
    'f1_score_macro': [f1]
})
df_metrics.to_csv('training_mlp_improved_metrics.csv', index=False)
print("✅ training_mlp_improved_metrics.csv saved")

# --- Predict on Test Set ---
print("Predicting on test set...")
pred_probs_test = model.predict(X_test_full)
pred_classes_test = np.argmax(pred_probs_test, axis=1)
pred_labels_test = target_encoder.inverse_transform(pred_classes_test)

# Ensure 'ID' column exists in the test set for output
if 'ID' in my_test.columns:
    output_df = my_test[['ID']].copy()
    output_df['Predicted_Credit_Score'] = pred_labels_test
    output_df.to_csv('predictions_mlp_improved.csv', index=False)
    print("✅ predictions_mlp_improved.csv saved")
else:
    print("Warning: 'ID' column not found in test data. Cannot create standard prediction output file.")
    # Save predictions without ID if necessary
    pd.DataFrame({'Predicted_Credit_Score': pred_labels_test}).to_csv('predictions_mlp_improved_noID.csv', index=False)
    print("✅ predictions_mlp_improved_noID.csv saved")