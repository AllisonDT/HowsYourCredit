import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader import CreditDataset
from transformer_model import TabularTransformer
from tqdm import tqdm
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
X_sample = np.load("X_train.npy")
INPUT_DIM = X_sample.shape[1]
NUM_CLASSES = 3

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# Load data
train_dataset = CreditDataset("X_train.npy", "y_train.npy")
val_dataset = CreditDataset("X_val.npy", "y_val.npy")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model
model = TabularTransformer(input_dim=INPUT_DIM, num_classes=NUM_CLASSES).to(device)

# Optional: compute class weights
class_counts = torch.tensor([13365, 21865, 40110], dtype=torch.float32)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum()
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Track best validation accuracy
best_val_acc = 0.0

# Training loop
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    for batch_X, batch_y in tqdm(train_loader, desc="Training", leave=False):
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == batch_y).sum().item()
        total += batch_y.size(0)

    train_acc = 100 * correct / total
    avg_loss = running_loss / len(train_loader)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_X, val_y in tqdm(val_loader, desc="Validating", leave=False):
            val_X, val_y = val_X.to(device), val_y.to(device)
            val_outputs = model(val_X)
            _, val_pred = torch.max(val_outputs, 1)
            val_correct += (val_pred == val_y).sum().item()
            val_total += val_y.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")

    # Save best model
    # if val_acc > best_val_acc:
    #     best_val_acc = val_acc
    #     torch.save(model.state_dict(), "best_model.pth")
    #     print(f"✅ Saved new best model at epoch {epoch+1} with Val Acc: {val_acc:.2f}%")
