import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CUB200Dataset
import time
import logging
from torch.utils.tensorboard import SummaryWriter


# --- 0. 定义注意力模块 (SELayer) ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze 操作: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 操作: 两层全连接层
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# --- 日志配置 ---
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('training_improved.log', mode='w')  # <-- 文件名已更改
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# --- TensorBoard配置 ---
writer = SummaryWriter('runs/improved_model')  # <-- 目录已更改
logging.info("TensorBoard日志将保存在 'runs/improved_model' 目录下")

# --- 设备配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

# --- 数据预处理和加载 (与之前相同) ---
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

# --- 加载预训练模型并嵌入注意力模块 ---
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 在ResNet的最后一个阶段(layer4)的末尾添加我们的SELayer
# ResNet-50的layer4输出通道数为2048
model.layer4.add_module("SELayer", SELayer(channel=2048))

# --- 解冻更多层进行微调 ---
# 我们只冻结前面的层，解冻layer4和fc层进行训练
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# 替换分类头
num_ftrs = model.fc.in_features
num_classes = 200
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# --- 定义损失函数和优化器 ---
criterion = nn.CrossEntropyLoss()
# 收集所有需要训练的参数
params_to_update = []
logging.info("需要训练的参数:")
for name, param in model.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        logging.info(f"\t{name}")
optimizer = optim.Adam(params_to_update, lr=0.001)

# --- 训练循环 (与之前相同) ---
num_epochs = 25
best_acc = 0.0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    logging.info(f'Epoch {epoch + 1}/{num_epochs}')
    logging.info('-' * 10)

    # ... (训练和验证循环代码与baseline完全相同，这里省略以保持简洁) ...
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
    writer.add_scalar('Loss/val', val_epoch_loss, epoch + 1)
    writer.add_scalar('Accuracy/val', val_epoch_acc, epoch + 1)

    # 保存最佳模型
    if val_epoch_acc > best_acc:
        best_acc = val_epoch_acc
        torch.save(model.state_dict(), 'improved_best_model.pth')  # <-- 文件名已更改
        logging.info(f"新最佳模型已保存！准确率: {best_acc:.4f}")

    epoch_time_elapsed = time.time() - epoch_start_time
    logging.info(f'本Epoch耗时 {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

logging.info(f'训练完成！最佳验证准确率: {best_acc:4f}')
writer.close()