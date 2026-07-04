import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import wandb

# =====================
# CONFIG
# =====================
DATA_DIR = ""          
BATCH_SIZE = 32
EPOCHS = 30
PATIENCE = 5           # Early stopping
LR = 1e-4
WEIGHT_DECAY = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if torch.cuda.is_available() else 0

# =====================
# INIT WANDB
# =====================
wandb.init(
    project="image-classification",   
    name="denseann-run",               
    config={
        "batch_size":      BATCH_SIZE,
        "epochs":          EPOCHS,
        "lr":              LR,
        "weight_decay":    WEIGHT_DECAY,
        "architecture":    "DenseANN",
        "optimizer":       "AdamW",
        "scheduler":       "CosineAnnealingLR",
        "label_smoothing": 0.1,
        "dropout_1":       0.4,
        "dropout_2":       0.3,
        "hidden_layers":   [1024, 512, 256],
        "input_size":      224 * 224 * 3,
        "device":          DEVICE,
        "patience":        PATIENCE,
        "batch_norm":      True,
    }
)
config = wandb.config   

print(f"Dispositivo in uso: {DEVICE}")

# =====================
# TRASFORMAZIONI
# =====================
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================
# LOADING AND SPLIT (Train, Valid, Test)
# =====================
train_dataset    = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
full_val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_test_transforms)

targets = full_val_dataset.targets
val_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.5,
    shuffle=True,
    stratify=targets,
    random_state=42
)

val_dataset  = Subset(full_val_dataset, val_idx)
test_dataset = Subset(full_val_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

NUM_CLASSES = len(train_dataset.classes)
print(f"Classes ({NUM_CLASSES}): {train_dataset.classes}")
print(f"Samples → Train: {len(train_dataset)} | Valid: {len(val_dataset)} | Test: {len(test_dataset)}")

# Log info dataset ON W&B
wandb.config.update({
    "num_classes":    NUM_CLASSES,
    "train_samples":  len(train_dataset),
    "val_samples":    len(val_dataset),
    "test_samples":   len(test_dataset),
})

# =====================
# MODEL
# =====================
class DenseANN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(224 * 224 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(config.dropout_1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(config.dropout_2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.net(self.flatten(x))

model = DenseANN(num_classes=NUM_CLASSES).to(DEVICE)

# Trace gradients and model weigths on W&B
wandb.watch(model, log="all", log_freq=50)

# =====================
# LOSS, OPTIMIZER, SCHEDULER
# =====================
criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)

# =====================
# TRAINING FUNCTIONS / VALIDATION
# =====================
def train_one_epoch(model, loader):
    model.train()
    running_loss, correct = 0.0, 0
    loop = tqdm(loader, desc="Training", leave=False)
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        loop.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader), correct / len(loader.dataset)


def validate(model, loader, desc="Validation"):
    model.eval()
    running_loss, correct = 0.0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return running_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels

# =====================
# TRAINING LOOP
# =====================
best_val_acc = 0.0
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

# =====================
# EARLY STOPPING
# =====================
best_val_loss    = float("inf")
patience_counter = 0

for epoch in range(config.epochs):
    current_lr = scheduler.get_last_lr()[0] if epoch > 0 else config.lr
    print(f"\nEpoch {epoch+1}/{config.epochs}  (LR: {current_lr:.2e})")

    train_loss, train_acc = train_one_epoch(model, train_loader)
    val_loss, val_acc, _, _ = validate(model, val_loader)

    scheduler.step()

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ── LOG EPOCH SU W&B ──────────────────────────────────────────────────────
    wandb.log({
        "epoch":      epoch + 1,
        "train/loss": train_loss,
        "train/acc":  train_acc,
        "val/loss":   val_loss,
        "val/acc":    val_acc,
        "lr":         current_lr,
    }, step=epoch + 1)
    # ─────────────────────────────────────────────────────────────────────────

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print(f"Model saved! Best Val Acc: {best_val_acc:.4f}")

        # Checkpoint save also on W&B
        wandb.save("best_model.pth")

    # ── EARLY STOPPING (based on val_loss) ──────────────────────────────────
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        print(f"Early stopping counter: {patience_counter}/{config.patience}")
        if patience_counter >= config.patience:
            print(f"\nEarly stopping enabled at epoch {epoch+1}. Val loss doesn't improve from {config.patience} epoch.")
            wandb.log({"early_stopping_epoch": epoch + 1})
            break
    # ─────────────────────────────────────────────────────────────────────────

print(f"\nTraining completed. Best Val Acc: {best_val_acc:.4f}")

# =====================
# GRAFICI: LOSS, ACCURACY, LR  (saved also on W&B)
# =====================
epochs_range = range(1, config.epochs + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="o")
axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Val Accuracy",   marker="o")
axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs_range, history["lr"], label="Learning Rate", marker="o", color="green")
axes[2].set_title("Learning Rate (Cosine Scheduler)"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
wandb.log({"charts/training_curves": wandb.Image("training_curves.png")})  # W&B
plt.show()

# =====================
# FINAL TEST AND CONFUSION MATRIX
# =====================
print("\n--- Final evaluation on TEST SET ---")
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

_, test_acc, y_pred, y_true = validate(model, test_loader, desc="Testing")
print(f"Accuracy on Test Set: {test_acc:.4f}")

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

# Log final metrics on W&B
wandb.log({
    "test/acc": test_acc,
    **{f"test/f1_{cls}": report[cls]["f1-score"] for cls in train_dataset.classes},
})

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(8, NUM_CLASSES), max(6, NUM_CLASSES - 2)))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)
plt.xlabel("Predizioni"); plt.ylabel("Valori Reali")
plt.title(f"Confusion Matrix - Test Set (Acc: {test_acc:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# Log confusion matrix on W&B (native interactive version)
wandb.log({
    "charts/confusion_matrix": wandb.Image("confusion_matrix.png"),
    "test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_true,
        preds=y_pred,
        class_names=train_dataset.classes
    )
})

plt.show()

# =====================
# CLOSE RUN W&B
# =====================
wandb.finish()
