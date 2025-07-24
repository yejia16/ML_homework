import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CUB200Dataset
import time
import logging
from torch.utils.tensorboard import SummaryWriter  # 1. 导入SummaryWriter

# --- 1.日志配置 ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('training_baseline.log', mode='w')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- 2. TensorBoard配置 ---
# 创建一个writer实例，日志将保存在 'runs/baseline_model' 目录下
writer = SummaryWriter('runs/baseline_model')
logging.info("TensorBoard日志将保存在 'runs/baseline_model' 目录下")

# --- 3. 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- 4. 数据预处理和加载 ---
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = './CUB_200_2011'
train_dataset = CUB200Dataset(root_dir=data_dir, train=True, transform=data_transforms['train'])
val_dataset = CUB200Dataset(root_dir=data_dir, train=False, transform=data_transforms['val'])

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
logging.info(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(val_dataset)}")

# --- 5. 加载预训练模型并修改 ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
num_classes = 200
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# --- (可选) 将模型图写入TensorBoard ---
# dataiter = iter(train_loader)
# images, _ = next(dataiter)
# writer.add_graph(model, images.to(device))

# --- 6. 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# --- 7. 训练循环 ---
num_epochs = 25
best_acc = 0.0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    logging.info(f'Epoch {epoch + 1}/{num_epochs}')
    logging.info('-' * 10)

    # --- 训练阶段 ---
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = running_corrects.double() / len(train_dataset)
    logging.info(f'训练 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f}')

    # 3. 记录训练损失和准确率到TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch + 1)

    # --- 验证阶段 ---
    model.eval()
    val_running_loss = 0.0
    val_running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item() * inputs.size(0)
            val_running_corrects += torch.sum(preds == labels.data)

    val_epoch_loss = val_running_loss / len(val_dataset)
    val_epoch_acc = val_running_corrects.double() / len(val_dataset)
    logging.info(f'验证 损失: {val_epoch_loss:.4f} 准确率: {val_epoch_acc:.4f}')

    # 4. 记录验证损失和准确率到TensorBoard
    writer.add_scalar('Loss/val', val_epoch_loss, epoch + 1)
    writer.add_scalar('Accuracy/val', val_epoch_acc, epoch + 1)

    # 保存最佳模型
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        torch.save(model.state_dict(), 'baseline_best_model.pth')
        logging.info(f"新最佳模型已保存！准确率: {best_acc:.4f}")

    epoch_time_elapsed = time.time() - epoch_start_time
    logging.info(f'本Epoch耗时 {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

logging.info(f'训练完成！最佳验证准确率: {best_acc:4f}')
# 5. 关闭writer
writer.close()