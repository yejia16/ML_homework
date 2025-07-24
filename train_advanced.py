import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import CUB200Dataset
import time
import logging
from torch.utils.tensorboard import SummaryWriter
import argparse  # 1. 导入argparse，用于处理命令行参数


def main(args):
    # --- 日志和TensorBoard配置 (根据模型动态命名) ---
    log_filename = f'training_{args.model}.log'
    tensorboard_dir = f'runs/{args.model}'

    logger = logging.getLogger()
    # 清除之前的handlers，防止日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(log_filename, mode='w')
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    writer = SummaryWriter(tensorboard_dir)
    logging.info(f"模型: {args.model}")
    logging.info(f"TensorBoard日志将保存在 '{tensorboard_dir}' 目录下")

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
    data_dir = args.data_dir
    train_dataset = CUB200Dataset(root_dir=data_dir, train=True, transform=data_transforms['train'])
    val_dataset = CUB200Dataset(root_dir=data_dir, train=False, transform=data_transforms['val'])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    logging.info(f"训练集样本数: {len(train_dataset)}, 测试集样本数: {len(val_dataset)}")

    # --- 2. 根据参数动态加载模型 ---
    num_classes = 200
    if args.model == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        # 定义需要差分学习率的参数组
        params_body = model.features.parameters()
        params_head = model.classifier.parameters()
    elif args.model == 'efficientnet_b3':
        model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        # 定义需要差分学习率的参数组
        params_body = model.features.parameters()
        params_head = model.classifier.parameters()
        # 在这里增加一个新的 elif 分支
    elif args.model == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        # ResNet没有独立的classifier，我们把fc层视为head，其余视为body
        params_to_update = []
        params_body = []
        params_head = []
        for name, param in model.named_parameters():
            if 'fc' in name:
                params_head.append(param)
            else:
                params_body.append(param)
    else:
        raise ValueError("不支持的模型，请选择 'densenet121' 或 'efficientnet_b3' 或 'resnet50'")

    # 全模型微调：解冻所有层
    for param in model.parameters():
        param.requires_grad = True

    model = model.to(device)

    # --- 3. 设置差分学习率优化器 ---
    # 使用AdamW，它在微调时通常比Adam表现更好
    optimizer = optim.AdamW([
        {'params': params_body, 'lr': args.lr_body},
        {'params': params_head, 'lr': args.lr_head}
    ])
    logging.info(f"优化器设置完毕：模型主体学习率 {args.lr_body}, 分类头学习率 {args.lr_head}")

    # --- 4. 设置余弦退火学习率调度器 ---
    # T_max是学习率周期的长度，这里设为总的epoch数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    logging.info(f"学习率调度器: 余弦退火, T_max={args.epochs}")

    criterion = nn.CrossEntropyLoss()

    # --- 训练循环 ---
    best_acc = 0.0
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        logging.info(f'Epoch {epoch + 1}/{args.epochs}')
        logging.info('-' * 10)

        # 训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        current_lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        logging.info(f'训练 损失: {epoch_loss:.4f} 准确率: {epoch_acc:.4f} 学习率: {current_lr:.7f}')
        writer.add_scalar('Loss/train', epoch_loss, epoch + 1)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch + 1)
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_dataset)
        logging.info(f'验证 损失: {val_epoch_loss:.4f} 准确率: {val_epoch_acc:.4f}')
        writer.add_scalar('Loss/val', val_epoch_loss, epoch + 1)
        writer.add_scalar('Accuracy/val', val_epoch_acc, epoch + 1)

        # 更新学习率
        scheduler.step()

        # 保存最佳模型
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            torch.save(model.state_dict(), f'{args.model}_best_model.pth')
            logging.info(f"新最佳模型已保存！准确率: {best_acc:.4f}")

        epoch_time_elapsed = time.time() - epoch_start_time
        logging.info(f'本Epoch耗时 {epoch_time_elapsed // 60:.0f}m {epoch_time_elapsed % 60:.0f}s')

    logging.info(f'训练完成！最佳验证准确率: {best_acc:4f}')
    writer.close()


if __name__ == '__main__':
    # 5. 定义所有可配置的命令行参数
    parser = argparse.ArgumentParser(description='高级模型训练脚本')
    parser.add_argument('--model', type=str, required=True, choices=['densenet121', 'efficientnet_b3','resnet50'],
                        help='要使用的模型架构')
    parser.add_argument('--epochs', type=int, default=20, help='训练的总轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr_head', type=float, default=1e-3, help='分类头的学习率')
    parser.add_argument('--lr_body', type=float, default=1e-4, help='模型主体的学习率')
    parser.add_argument('--data_dir', type=str, default='./CUB_200_2011', help='数据集的路径')

    args = parser.parse_args()
    main(args)