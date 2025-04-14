import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import cohen_kappa_score

df1 = pd.read_csv('mlp/predictions_mlp.csv')
df2 = pd.read_csv('cnn/predictions_cnn.csv')
df3 = pd.read_csv('lstm/predictions_lstm.csv')
df4 = pd.read_csv('transformers/predictions_transformer.csv')

df1.rename(columns={'Predicted_Credit_Score': 'MLP'}, inplace=True)
df2.rename(columns={'Predicted_Credit_Score': 'CNN'}, inplace=True)
df3.rename(columns={'Predicted_Credit_Score': 'LSTM'}, inplace=True)
df4.rename(columns={'Predicted_Credit_Score': 'Transformers'}, inplace=True)

df = df1.merge(df2, on='ID').merge(df3, on='ID').merge(df4, on='ID')

models = ['MLP', 'CNN', 'LSTM', 'Transformers']
all_ratings = sorted(set(df['MLP']) | set(df['CNN']) | set(df['LSTM']) | set(df['Transformers']))

counts = {}
for model in models:
    counts[model] = df[model].value_counts().reindex(all_ratings, fill_value=0)

x = np.arange(len(all_ratings))
width = 0.2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16))

# Plot the distribution of credit ratings
ax1.bar(x - 1.5 * width, counts['MLP'], width, label='MLP')
ax1.bar(x - 0.5 * width, counts['CNN'], width, label='CNN')
ax1.bar(x + 0.5 * width, counts['LSTM'], width, label='LSTM')
ax1.bar(x + 1.5 * width, counts['Transformers'], width, label='Transformers')
ax1.set_xticks(x)
ax1.set_xticklabels(all_ratings)
ax1.set_xlabel('Credit Rating')
ax1.set_ylabel('Count')
ax1.set_title('Distribution of Credit Rating Predictions')
ax1.legend()

num_models = len(models)
kappa_matrix = np.zeros((num_models, num_models))

# Compute pairwise Cohen's kappa scores between models
for i, m1 in enumerate(models):
    for j, m2 in enumerate(models):
        if i == j:
            kappa_matrix[i, j] = 1.0
        else:
            kappa_matrix[i, j] = cohen_kappa_score(df[m1], df[m2])

cax = ax2.imshow(kappa_matrix, cmap='coolwarm', vmin=0, vmax=1)

ax2.set_xticks(np.arange(num_models))
ax2.set_xticklabels(models)
ax2.set_yticks(np.arange(num_models))
ax2.set_yticklabels(models)
ax2.set_title("Pairwise Cohen's Kappa Scores Among Models")

for i in range(num_models):
    for j in range(num_models):
        ax2.text(j, i, f"{kappa_matrix[i, j]:.2f}", ha="center", va="center", color="black")

fig.colorbar(cax, ax=ax2, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
