# ------------------------- imports -------------------------
import os, itertools
from pathlib import Path
import torch, torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# ONNX to TensorFlow to H5
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# ------------------------- config --------------------------
class CFG:
    data_dir     = Path("eye_data")
    img_size     = 224
    batch_size   = 32
    n_epochs     = 50
    lr           = 1e-4
    num_classes  = 5
    class_names  = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'macular_degeneration', 'normal']
    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = CFG()

# -------------------- transforms / dataloaders -------------
train_tf = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.1, 0.1, 0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])
val_tf = transforms.Compose([
    transforms.Resize((cfg.img_size, cfg.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

train_dir = r'D:\Nazia Apu\Data_nazia-20250516T152449Z-1-001\Data_nazia\Train'
val_dir   = r'D:\Nazia Apu\Data_nazia-20250516T152449Z-1-001\Data_nazia\Val'

train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

cfg.class_names = train_ds.classes

train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

# ------------------------- model ---------------------------
class EyeDiseaseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*14*14, 512),
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

model = EyeDiseaseCNN(cfg.num_classes).to(cfg.device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, verbose=True)

best_val_acc = 0
saved_confusion_matrices = []  # Store (epoch_num, confusion_matrix)

# ------------------------- training loop -------------------
for epoch in range(cfg.n_epochs):
    model.train()
    tr_loss, tr_correct = 0, 0
    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.n_epochs}"):
        imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * imgs.size(0)
        tr_correct += (outputs.argmax(1) == labels).sum().item()

    tr_loss /= len(train_ds)
    tr_acc = tr_correct / len(train_ds)

    model.eval()
    val_correct = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(cfg.device), labels.to(cfg.device)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            val_correct += (preds == labels).sum().item()
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    val_acc = val_correct / len(val_ds)
    scheduler.step(val_acc)

    print(f"Epoch {epoch+1:02d} | TrainLoss {tr_loss:.4f} | TrainAcc {tr_acc:.4f} | ValAcc {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_eye_disease_cnn.pth")
        print("✅ Saved best model")

        print("\n--- Validation classification report (best so far) ---")
        print(classification_report(y_true, y_pred, target_names=cfg.class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)
        print("Confusion matrix:\n", cm)
        saved_confusion_matrices.append((epoch+1, cm))

print(f"\n✅ Training finished. Best validation accuracy: {best_val_acc:.4f}")

# ----------------- Plot all confusion matrices at the end -----------------
for epoch_num, cm in saved_confusion_matrices:
    print(f"\n📊 Confusion Matrix for Epoch {epoch_num}")
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(np.arange(cfg.num_classes))
    ax.set_yticks(np.arange(cfg.num_classes))
    ax.set_xticklabels(cfg.class_names, rotation=45, ha="right")
    ax.set_yticklabels(cfg.class_names)
    for i in range(cfg.num_classes):
        for j in range(cfg.num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "black")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion Matrix - Epoch {epoch_num}")
    plt.tight_layout()
    plt.show()

# ------------------- Export to ONNX + H5 -------------------
print("\n🔄 Exporting PyTorch model to ONNX...")

dummy_input = torch.randn(1, 3, cfg.img_size, cfg.img_size).to(cfg.device)
torch.onnx.export(model, dummy_input, "eye_disease_cnn.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=11)

print("✅ Saved as ONNX: eye_disease_cnn.onnx")

print("🔄 Converting ONNX → TensorFlow → .h5...")
onnx_model = onnx.load("eye_disease_cnn.onnx")
tf_rep = prepare(onnx_model)

# Optional: Export as TensorFlow SavedModel
tf_rep.export_graph("eye_disease_tf_model")

# ✅ Correct Keras-compatible model for .h5
keras_model = tf_rep.keras_model
keras_model.save("eye_disease_model.h5")
print("✅ Saved as H5: eye_disease_model.h5")

# ------------------------- inference helper -----------------
def predict_eye_image(img_path, model_path="best_eye_disease_cnn.pth"):
    from PIL import Image
    tfm = val_tf
    model = EyeDiseaseCNN(cfg.num_classes).to(cfg.device)
    model.load_state_dict(torch.load(model_path, map_location=cfg.device))
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(cfg.device)
    with torch.no_grad():
        pred = model(x).argmax(1).item()
    return cfg.class_names[pred]
