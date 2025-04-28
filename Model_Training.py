import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm


IMG_SIZE = 224
SEQ_LEN = 5
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATIENCE = 3  # Number of epochs to wait for improvement before stopping early

class SequenceDataset(Dataset):
    def __init__(self, sequences, labels, transform=None):
        self.sequences = sequences
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]

        frames_tensor = torch.stack([
            self.transform(Image.open(frame_path).convert("RGB"))
            for frame_path in sequence
        ])
        return frames_tensor, torch.tensor(label, dtype=torch.float32)

def create_sequences(data_dir, seq_len=5):
    sequences = []
    labels = []
    class_map = {"notdrowsy": 0, "drowsy": 1}

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        files = sorted(os.listdir(class_path))  # simulate time order
        paths = [os.path.join(class_path, f) for f in files]
        for i in range(len(paths) - seq_len + 1):
            sequences.append(paths[i:i+seq_len])
            labels.append(class_map[class_name])
    return sequences, labels

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

sequences, labels = create_sequences("train data", SEQ_LEN)
train_seq, val_seq, train_lbl, val_lbl = train_test_split(sequences, labels, test_size=0.2, stratify=labels, random_state=42)

train_dataset = SequenceDataset(train_seq, train_lbl, transform)
val_dataset = SequenceDataset(val_seq, val_lbl, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


class DrowsinessModel(nn.Module):
    def __init__(self, feature_dim=1280, hidden_size=64):
        super().__init__()
        base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        self.cnn = nn.Sequential(*list(base_model.children())[:-2])  # remove head
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.rnn = nn.GRU(input_size=feature_dim, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        x = self.cnn(x)
        x = self.pool(x).view(B, T, -1)
        _, h = self.rnn(x)
        out = self.classifier(h[-1])
        return out.squeeze()

model = DrowsinessModel().to(DEVICE)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend((preds > 0.5).cpu().numpy())
    
    # Classification Report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Not Drowsy', 'Drowsy']))
    
    # Precision and Recall for each class
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

train_losses, val_losses = [], []
best_val_loss = float('inf')
epochs_without_improvement = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_losses.append(running_loss / len(train_loader))

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = criterion(preds, y)
            val_loss += loss.item()
    val_losses.append(val_loss / len(val_loader))
    print(f"Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")

    # Early Stopping
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]
        epochs_without_improvement = 0
        # Save the model with the best validation loss
        torch.save(model.state_dict(), "best_drowsiness_model.pt")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print(f"Early stopping after {epoch+1} epochs")
        break

evaluate(model, val_loader)


plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

# Save the plot as an image
plt.savefig("training_loss_plot.png")  
plt.close()  # 

torch.save(model.state_dict(), "final_drowsiness_model.pt")
