import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# Проверка доступности GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Загрузка данных
dirname = "/home/bektemir/Desktop/my_projects/faceRecognation/dataset"
data_files = os.listdir(dirname)
data = []
labels = []

for file in data_files:
    with open(dirname + '/' + file, 'r') as f:
        data_array = np.array([list(map(float, line.split())) for line in f.readlines()])
        data.append(data_array)
        labels.append(1 if file.split('-')[0] == file.split('-')[1] else 0)

data = np.array(data)
labels = np.array(labels)

# Разделение данных на обучающий и тестовый наборы
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape labels
train_labels = train_labels.reshape((-1, 1))
test_labels = test_labels.reshape((-1, 1))

# PyTorch Model
class ModelPT(nn.Module):
    def __init__(self):
        super(ModelPT, self).__init__()
        self.fc1 = nn.Linear(18 * 17, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Создание модели и перемещение ее на GPU
model_pt = ModelPT().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model_pt.parameters())

# Конвертация NumPy массивов в PyTorch тензоры и перемещение на GPU
train_data_pt = torch.from_numpy(train_data).float().to(device)
train_labels_pt = torch.from_numpy(train_labels).float().to(device)

# Создание DataLoader для PyTorch
train_dataset = TensorDataset(train_data_pt, train_labels_pt)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Обучение модели на GPU
for epoch in range(10):
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model_pt(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Save checkpoint at the end of each epoch
    checkpoint_path_pt = f"checkpoints/model_checkpoint_epoch_pt_{epoch:02d}.pth"
    torch.save(model_pt.state_dict(), checkpoint_path_pt)

    # Оценка модели на тестовом наборе данных
    model_pt.eval()

    # Конвертация тестовых данных в PyTorch тензор и перемещение на GPU
    test_data_pt = torch.from_numpy(test_data).float().to(device)
    test_labels_pt = torch.from_numpy(test_labels).float().to(device)

    with torch.no_grad():
        outputs_pt = model_pt(test_data_pt)
        predicted_labels_pt = (torch.sigmoid(outputs_pt) > 0.5).float()

    # Рассчет точности на GPU
    accuracy_pt = (predicted_labels_pt == test_labels_pt).float().mean().item()
    print(f'Epoch {epoch + 1}, Test Accuracy (PyTorch on GPU): {accuracy_pt}')

    # Set the model back to training mode
    model_pt.train()


# Оценка модели на тестовом наборе данных
model_pt.eval()

# Конвертация тестовых данных в PyTorch тензор и перемещение на GPU
test_data_pt = torch.from_numpy(test_data).float().to(device)
test_labels_pt = torch.from_numpy(test_labels).float().to(device)

with torch.no_grad():
    outputs_pt = model_pt(test_data_pt)
    predicted_labels_pt = (torch.sigmoid(outputs_pt) > 0.5).float()

# Рассчет точности на GPU
accuracy_pt = (predicted_labels_pt == test_labels_pt).float().mean().item()
print(f'Test Accuracy (PyTorch on GPU): {accuracy_pt}')
