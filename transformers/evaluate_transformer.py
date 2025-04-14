import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from transformer_model import TabularTransformer

# Load pre-saved validation sets
X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_tensor = torch.tensor(X_val).float().to(device)
y_tensor = torch.tensor(y_val).long().to(device)

# Load model
input_dim = X_val.shape[1]
model = TabularTransformer(input_dim=input_dim)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

# Evaluate
with torch.no_grad():
    outputs = model(X_tensor)
    preds = torch.argmax(outputs, dim=1)

# Calculate Metrics
val_loss = nn.CrossEntropyLoss()(outputs, y_tensor).item()
val_acc = (preds == y_tensor).sum().item() / len(y_tensor)

cm = confusion_matrix(y_tensor.cpu(), preds.cpu())
tp = np.diag(cm)
fp = cm.sum(axis=0) - tp
fn = cm.sum(axis=1) - tp
tn = cm.sum() - (fp + fn + tp)

precision = precision_score(y_tensor.cpu(), preds.cpu(), average='macro')
recall = recall_score(y_tensor.cpu(), preds.cpu(), average='macro')
f1 = f1_score(y_tensor.cpu(), preds.cpu(), average='macro')

# Save metrics
df_metrics = pd.DataFrame({
    "val_accuracy": [val_acc],
    "val_loss": [val_loss],
    "cm_tp": [tp.sum()],
    "cm_fn": [fn.sum()],
    "cm_fp": [fp.sum()],
    "cm_tn": [tn.sum()],
    "precision": [precision],
    "recall": [recall],
    "f1_score": [f1]
})

df_metrics.to_csv("training_transformer.csv", index=False)
print("✅ Metrics saved to training_transformer.csv")
