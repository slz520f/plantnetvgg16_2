import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import json
import os
from tqdm import tqdm




# データセットのパス
train_data_dir = '/Users/mame/Downloads/plantnet_300K/images/train_2'
validation_data_dir = '/Users/mame/Downloads/plantnet_300K/images/validation_2'
test_data_dir = '/Users/mame/Downloads/plantnet_300K/images/test_2'

# ハイパーパラメータ
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3
INITIAL_LEARNING_RATE = 0.001

# データ拡張の設定
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# データセットの読み込み
train_dataset = ImageFolder(train_data_dir, transform=train_transform)
val_dataset = ImageFolder(validation_data_dir, transform=val_transform)
test_dataset = ImageFolder(test_data_dir, transform=val_transform)

# データローダーの設定
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# クラス数の設定
num_classes = len(train_dataset.classes)

# モデルの構築
model = models.efficientnet_b4(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# デバイスの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 損失関数とオプティマイザーの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=INITIAL_LEARNING_RATE)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

if __name__ == '__main__':# 訓練ループ
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # 検証
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)
        
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# モデルの保存
os.makedirs('output', exist_ok=True)
torch.save(model.state_dict(), 'output/plantnet_300k_model.pth')

# クラスインデックスの保存
class_to_idx = train_dataset.class_to_idx
with open('output/class_indices.json', 'w') as f:
    json.dump(class_to_idx, f)

# テストデータの評価
model.eval()
test_correct = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        test_correct += predicted.eq(labels).sum().item()

test_acc = test_correct / len(test_loader.dataset)
print(f"Test accuracy: {test_acc:.4f}")

# top-5 accuracyとaverage-5 accuracyの計算
def top_k_accuracy(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return correct[:k].any(dim=0).float().mean().item()

def average_k_accuracy(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    threshold = output.gather(1, pred[:, k-1].unsqueeze(1))
    return (output >= threshold).float().mul(target.float()).sum(1).gt(0).float().mean().item()

top_5_acc = 0
avg_5_acc = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        top_5_acc += top_k_accuracy(outputs, labels, k=5)
        avg_5_acc += average_k_accuracy(outputs, labels, k=5)

top_5_acc /= len(test_loader)
avg_5_acc /= len(test_loader)

print(f"Top-5 accuracy: {top_5_acc:.4f}")
print(f"Average-5 accuracy: {avg_5_acc:.4f}")

# 結果の保存
results = {
    "test_accuracy": test_acc,
    "top_5_accuracy": top_5_acc,
    "average_5_accuracy": avg_5_acc
}

with open('output/evaluation_results.json', 'w') as f:
    json.dump(results, f)

print("Training and evaluation completed. Results saved in the 'output' directory.")

if __name__ == '__main__':
    pass
