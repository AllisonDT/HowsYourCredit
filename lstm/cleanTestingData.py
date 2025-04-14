import pandas as pd
import numpy as np
import re

def convert_credit_history_age(age_str):
    """
    Convert a string like '22 Years and 1 Months' into a float (years).
    Returns NaN if the format doesn't match.
    """
    if pd.isna(age_str):
        return np.nan
    pattern = r'(\d+)\s*Years?\s*and\s*(\d+)\s*Months?'
    match = re.match(pattern, age_str)
    if match:
        years = int(match.group(1))
        months = int(match.group(2))
        return years + months / 12.0
    return np.nan

def clean_data(df):
    """
    Cleans the dataframe by:
    - Replacing placeholder values (e.g. underscores) with NaN.
    - Stripping whitespace from column names and string data.
    - Converting columns expected to be numeric.
    - Converting 'Credit_History_Age' to a numeric value.
    - Removing rows with unrealistic values (e.g. negative ages).
    """
    # Replace cells that consist entirely of underscores with NaN.
    df.replace(r'^\s*_+\s*$', np.nan, regex=True, inplace=True)
    
    # Trim whitespace from column names.
    df.columns = df.columns.str.strip()
    
    # Trim whitespace in string columns.
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].str.strip()
    
    # List of columns that should be numeric.
    numeric_cols = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts',
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date',
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries',
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month',
        'Amount_invested_monthly'
    ]
    
    # Convert specified columns to numeric, coercing errors to NaN.
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Convert 'Credit_History_Age' to a numeric value (in years).
    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(convert_credit_history_age)
    
    # Remove rows with non-positive ages (if age is an important feature).
    if 'Age' in df.columns:
        df = df[df['Age'] > 0]
    
    return df

if __name__ == "__main__":
    # Specify the path to your input CSV file.
    input_csv = "/home/jamjorj/Documents/python/credit/test.csv"  # Replace with your actual CSV filename.
    
    # Load the CSV data into a DataFrame.
    df = pd.read_csv(input_csv)
    print("Initial data shape:", df.shape)
    
    # Clean the data.
    df_clean = clean_data(df)
    print("Cleaned data shape:", df_clean.shape)
    
    # Save the cleaned data to a new CSV file.
    output_csv = "testCleaned.csv"
    df_clean.to_csv(output_csv, index=False)
    print(f"Cleaned data saved to {output_csv}")
