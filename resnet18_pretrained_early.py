import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import wandb

# =====================
# CONFIG
# =====================
DATA_DIR = ""          # Modify with your own path
BATCH_SIZE = 32
EPOCHS = 15
PATIENCE = 5           # Early stopping: ferma se val_loss non migliora per N epoch
LR = 1e-4
WEIGHT_DECAY = 1e-4     # Penalize the weight that are too big
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if torch.cuda.is_available() else 0     #Number of parallel process which will be use for training

# =====================
# INIT WANDB
# =====================
wandb.init(     #Initialize Weight & Bias
    project="image-classification",   # Name of the project W&B
    name="resnet18-pretrained-run",               # Name of the run
    config={
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "lr":            LR,
        "weight_decay":  WEIGHT_DECAY,
        "architecture":  "ResNet18",
        "optimizer":     "AdamW",       #Algorithm which update the weights
        "scheduler":     "CosineAnnealingLR",   #Modify the learning rate during the training
        "label_smoothing": 0.1,     # Permit that the prediction probabilty isn't 100% on one class but dispatch'
        "dropout":       0.3,       # Turn off randomly 30% of the neurons for avoid overlearning
        "device":        DEVICE,
        "patience":      PATIENCE,
    }
)
config = wandb.config   # access the central parameter

print(f"Dispositivo in uso: {DEVICE}")

# =====================
# TRANSFORMATION
# =====================
train_transforms = transforms.Compose([     #Used for training
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),   #Cut randomly a part of the image which is between 75 and 100% of the initial size of the image and it's resized on 224x224 pixels'
    transforms.RandomHorizontalFlip(),  # Flip horizontaly the image with a probability of 50%
    transforms.RandomRotation(15),     # Turn randomly the image between -15° and 15°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),   # Modify randomly the colors in fonction of the parameters
    transforms.ToTensor(),  #Convert the image in PyTorch tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  #Normalize the pixels for fit to ImageNet
])

val_test_transforms = transforms.Compose([      #Used for validation and test
    transforms.Resize((224, 224)),      #The images are just resized
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# =====================
# CARICAMENTO E SPLIT (Train, Valid, Test)
# =====================
train_dataset    = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
full_val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "valid"), transform=val_test_transforms)

targets = full_val_dataset.targets      #Take the labels from the images
val_idx, test_idx = train_test_split(
    np.arange(len(targets)),
    test_size=0.5,      #50% validation/50% test
    shuffle=True,       #Shuffle before split
    stratify=targets,   #Permit that each split have the same class proportion
    random_state=42     #Permit to have the same split at each computation
)

val_dataset  = Subset(full_val_dataset, val_idx)
test_dataset = Subset(full_val_dataset, test_idx)

train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True)     #Load data by batch, pin_memory=True : optimization GPU
val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)     #No shuffle because we want stable results
test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

NUM_CLASSES = len(train_dataset.classes)    #train_dataset_.classes give the names of the classes in a list
print(f"Classes ({NUM_CLASSES}): {train_dataset.classes}")
print(f"Samples → Train: {len(train_dataset)} | Valid: {len(val_dataset)} | Test: {len(test_dataset)}")

# Load the dataset information on W&B
wandb.config.update({
    "num_classes":    NUM_CLASSES,
    "train_samples":  len(train_dataset),   #Give the size of the datasets
    "val_samples":    len(val_dataset),
    "test_samples":   len(test_dataset),
})

# =====================
# MODEL
# =====================
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)    #Load ResNet18 pre-trained, DEFAULT mean that the model is pre-trained

num_ftrs = model.fc.in_features     #Take the size of the last layer of ResNet
model.fc = nn.Sequential(           #Replace the last layer because it's trained for 1000 classes and we just want NUM_CLASSES classes
    nn.Dropout(p=config.dropout),   #Activate some Dropout during the training
    nn.Linear(num_ftrs, NUM_CLASSES)        #New last layer of the size of NUM_CLASSES
)
model = model.to(DEVICE)        #Send the model to the GPU cuda if available or on the CPU


wandb.watch(model, log="all", log_freq=50)      #Weight and Bias begin to watch the model, log="all" permit to record gradients, weight, histograms each 50 steps (log_freq=50)

# =====================
# LOSS, OPTIMIZER, SCHEDULER
# =====================
criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)     #Define loss fonction
optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)     #Define how the model learn
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)  #Permit to modify the learning rate during the training, T_max : duration of the decrease of the learning rate (here all the epoch), eta_min = min learning rate

# =====================
# TRAINING / VALIDATION FUNCTION
# =====================
def train_one_epoch(model, loader):
    model.train()           #Put the model in training mode, permit to activate the dropout for example
    running_loss, correct = 0.0, 0      #running_loss increment the loss of the batches, correct count the number of correct prediction
    loop = tqdm(loader, desc="Training", leave=False)   #Create a progress bar
    for images, labels in loop:
        images, labels = images.to(DEVICE), labels.to(DEVICE)   #Send the data to the device (cude or CPU)
        optimizer.zero_grad()       #Put the gradients to 0 for don't count the previous gradients
        outputs = model(images)     #Launch the network
        loss = criterion(outputs, labels)   #Compare the prediction with the real answer for calculate a loss
        loss.backward()         #Backpropagation for target the weight responsible of the loss
        optimizer.step()        #Modify the weights
        running_loss += loss.item()     #We use .item() for take the value of the tensor loss
        _, preds = torch.max(outputs, 1)    #Collect the predictions
        correct += (preds == labels).sum().item()       #Count the good predictions
        loop.set_postfix(loss=f"{loss.item():.4f}")     #Update the progress bar
    return running_loss / len(loader), correct / len(loader.dataset)    #Return the average loss and the accuracy


def validate(model, loader, desc="Validation"):
    model.eval()        #Put the model in the evaluation mode
    running_loss, correct = 0.0, 0
    all_preds, all_labels = [], []      #Store all the predictions and labels for the confusion matrix, preicsion and F1-score
    with torch.no_grad():       #Permit to not calculate the gradient and accelerate the calculation
        for images, labels in tqdm(loader, desc=desc, leave=False):     
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            all_preds.extend(preds.cpu().numpy())       #Save the predictions
            all_labels.extend(labels.cpu().numpy())     #Save the labels
    return running_loss / len(loader), correct / len(loader.dataset), all_preds, all_labels     #Return the average loss, the accuracy, all the predictions, all the labels

# =====================
# TRAINING LOOP
# =====================
best_val_acc = 0.0      #Store the best accuracy value
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}      #Dict for store the evolution of the metrics epoch after epoch

# =====================
# EARLY STOPPING
# =====================
best_val_loss   = float("inf")      #Initialize the loss to the infinite
patience_counter = 0            #Counter of epoch without improvement

for epoch in range(config.epochs):
    current_lr = scheduler.get_last_lr()[0] if epoch > 0 else config.lr     #Load the last value of lr
    print(f"\nEpoch {epoch+1}/{config.epochs}  (LR: {current_lr:.2e})")     #Print the number of the current epoch and the value of the lr

    train_loss, train_acc = train_one_epoch(model, train_loader)    #Call the training function
    val_loss, val_acc, _, _ = validate(model, val_loader)           #Evaluate the model

    scheduler.step()        #Update the scheduler which modify the learning rate for the next step

    history["train_loss"].append(train_loss)        #Store the result of the current epoch
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

    # ── LOG EPOCH ON W&B ──────────────────────────────────────────────────────
    wandb.log({             
        "epoch":      epoch + 1,
        "train/loss": train_loss,
        "train/acc":  train_acc,
        "val/loss":   val_loss,
        "val/acc":    val_acc,
        "lr":         current_lr,
    }, step=epoch + 1)
    # ─────────────────────────────────────────────────────────────────────────

    if val_acc > best_val_acc:      #Check if we the accuracy of the current epoch is the best
        best_val_acc = val_acc      #Update the best accuracy value
        torch.save(model.state_dict(), "best_model.pth")        #Save the weights of the model
        print(f"  ✔ Modello salvato! Miglior Val Acc: {best_val_acc:.4f}")

        # Salva il checkpoint anche su W&B
        wandb.save("best_model.pth")        #Load the weights in Weight and Bias

    # ── EARLY STOPPING ──────────────────────────────────
    if val_loss < best_val_loss:        #If best loss
        best_val_loss = val_loss        #Update the best loss value
        patience_counter = 0            #Put back the patience counter to 0
    else:                               #Otherwise
        patience_counter += 1           #Increment the counter
        print(f"  ⚠ Early stopping counter: {patience_counter}/{config.patience}")
        if patience_counter >= config.patience:     #Verify if the early stopping limit is reach
            print(f"\n⛔ Early stopping attivato all'epoch {epoch+1}. Val loss non migliora da {config.patience} epoch.")
            wandb.log({"early_stopping_epoch": epoch + 1})      #Save in weight and bias the epoch where we stop
            break                                               #Leave the loop for stop the training
    # ─────────────────────────────────────────────────────────────────────────

print(f"\nTraining completato. Miglior Val Acc: {best_val_acc:.4f}")

# =====================
# GRAPH: LOSS, ACCURACY, LR
# =====================
epochs_range = range(1, config.epochs + 1)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))     #Create a figure with 1 line and 3 columns, figsize define the size in inches

#Display the loss graph in the first column
axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", marker="o")
axes[0].plot(epochs_range, history["val_loss"],   label="Val Loss",   marker="o")
axes[0].set_title("Loss per Epoch"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].legend(); axes[0].grid(True, alpha=0.3)     #Display the label of each curve; display a grid

#Display the accuracy graph in the second column
axes[1].plot(epochs_range, history["train_acc"], label="Train Accuracy", marker="o")
axes[1].plot(epochs_range, history["val_acc"],   label="Val Accuracy",   marker="o")
axes[1].set_title("Accuracy per Epoch"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

#Display the learning rate graph in the third column
axes[2].plot(epochs_range, history["lr"], label="Learning Rate", marker="o", color="green")
axes[2].set_title("Learning Rate (Cosine Scheduler)"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("LR")
axes[2].set_yscale("log"); axes[2].legend(); axes[2].grid(True, alpha=0.3)      #Use a logarithmic scale because of the very low values

plt.tight_layout()      #Avoid that the title, label, legend overlap each other
plt.savefig("training_curves.png", dpi=150)     #Save the figure, dpi is the resolution of the picture
wandb.log({"charts/training_curves": wandb.Image("training_curves.png")})  # W&B
plt.show()      #Display the picture on the screen

# =====================
# FINAL TEST AND CONFUSION MATRIX
# =====================
print("\n--- Valutazione Finale sul TEST SET ---")
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))    #Load the file of the best model saved during the training, map_location=DEVICE load the weighs on the device, load_state_dict load the weight on the model

_, test_acc, y_pred, y_true = validate(model, test_loader, desc="Testing")  #Evaluate on the test dataset
print(f"Accuratezza sul Test Set: {test_acc:.4f}")

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=train_dataset.classes, output_dict=True)    #Create a detailled report, output_dict give us a python dict, without that it would have been just text
print(classification_report(y_true, y_pred, target_names=train_dataset.classes))

# Log metriche finali su W&B
wandb.log({
    "test/acc": test_acc,
    **{f"test/f1_{cls}": report[cls]["f1-score"] for cls in train_dataset.classes},
})      #Send the metrics and the evaluation on weight and bias

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(max(8, NUM_CLASSES), max(6, NUM_CLASSES - 2)))
sns.heatmap(                                #Display the confusion matrix
    cm, annot=True, fmt="d", cmap="Blues",  #annot = display number in each square, fmt=each value is integer
    xticklabels=train_dataset.classes,
    yticklabels=train_dataset.classes
)
plt.xlabel("Predizioni"); plt.ylabel("Valori Reali")
plt.title(f"Confusion Matrix - Test Set (Acc: {test_acc:.4f})")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

# Log confusion matrix su W&B (versione interattiva nativa)
wandb.log({
    "charts/confusion_matrix": wandb.Image("confusion_matrix.png"),
    "test/confusion_matrix": wandb.plot.confusion_matrix(       #Create a matrix in W&B
        y_true=y_true,
        preds=y_pred,
        class_names=train_dataset.classes
    )
})

plt.show()

# =====================
# CHIUDI RUN W&B
# =====================
wandb.finish()      #Close W&B, send the last logs, free the resources
