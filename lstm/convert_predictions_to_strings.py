
import pandas as pd

# Load the predictions
df = pd.read_csv("test_predictions.csv")

# Define mapping from numeric label to string
label_map = {
    0: "Bad",
    1: "Good",
    2: "Standard"
}

# Convert if necessary (ensure column is int first)
df['Predicted_Credit_Score'] = df['Predicted_Credit_Score'].astype(int).map(label_map)

# Save updated CSV
df.to_csv("test_predictions.csv", index=False)
print("Updated predictions saved to 'test_predictions_readable.csv'")
