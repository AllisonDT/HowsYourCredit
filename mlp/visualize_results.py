import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Plot accuracy & loss over epochs
epoch_df = pd.read_csv("./mlp/epoch_metrics.csv")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epoch_df['epoch'], epoch_df['train_loss'], label='Train Loss')
plt.plot(epoch_df['epoch'], epoch_df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss per Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epoch_df['epoch'], epoch_df['train_accuracy'], label='Train Acc')
plt.plot(epoch_df['epoch'], epoch_df['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend()
plt.tight_layout()
plt.savefig("epoch_plots.png")
plt.show()

# 2. Confusion matrix
cm = pd.read_csv("./mlp/confusion_matrix.csv").values
labels = ['Poor', 'Standard', 'Good']  # Adjust if using different encoding
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

# 3. Classification report bar chart
report_df = pd.read_csv("./mlp/per_class_report.csv").drop(['accuracy', 'macro avg', 'weighted avg', 'support'], errors='ignore')
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar')
plt.title('Per-Class Metrics')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("per_class_metrics.png")
plt.show()




