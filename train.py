import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader
import os

# 1. Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan device: {device}")

# 2. Transformasi data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# 3. Load Dataset
data_dir = "dataset-resized"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"Folder '{data_dir}' tidak ditemukan.")

train_data = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# 4. Load pretrained model ResNet18
model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, 6)  # TrashNet = 6 kelas
model = model.to(device)

# 5. Loss dan Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 6. Training Loop
epochs = 5
model.train()

for epoch in range(epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

# 7. Simpan Model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/trashnet_cnn.pth")
print("Model berhasil disimpan")
