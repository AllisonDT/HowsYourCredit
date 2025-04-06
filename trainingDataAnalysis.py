import pandas as pd

# Load the training data
train_data = pd.read_csv('trainCleaned.csv')

# Count the frequency of each credit score
credit_score_counts = train_data['Credit_Score'].value_counts().sort_index()

print("Frequency Distribution of Credit Scores:")
print(credit_score_counts)
