import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def train_one_epoch(model, data_loader, optimizer, criterion, device):
    """
    单个训练周期的实现（多模态版）。
    """
    model.train()
    running_loss = 0.0

    # 注意这里：解包出 3 个变量 (图像, 物理特征, 标签)
    for images, phys_feats, labels in tqdm(data_loader, desc="Training", leave=False):
        # 将所有数据搬运到 GPU/CPU
        images = images.to(device)
        phys_feats = phys_feats.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()  # 清零梯度

        # 前向传播：同时传入图像和物理特征
        outputs = model(images, phys_feats)

        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新权重

        running_loss += loss.item()

    avg_loss = running_loss / len(data_loader)
    return avg_loss


def validate(model, data_loader, criterion, device):
    """
    验证模型性能（多模态版）。
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        # 注意这里：解包出 3 个变量
        for images, phys_feats, labels in tqdm(data_loader, desc="Validation", leave=False):
            images = images.to(device)
            phys_feats = phys_feats.to(device)
            labels = labels.to(device)

            # 前向传播：同时传入图像和物理特征
            outputs = model(images, phys_feats)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 计算准确率
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / len(data_loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, epochs, lr, device, save_path):
    """
    训练和验证模型。
    """
    criterion = nn.CrossEntropyLoss()  # 分类任务的损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        # 训练
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Training Loss: {train_loss:.4f}")

        # 验证
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.2f}%")

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")


# 测试代码：直接运行开始训练
if __name__ == "__main__":
    from model import FruitClassifier
    from data_loader import get_data_loaders

    # 超参数设置
    EPOCHS = 40
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32  # 显存不够可以调小到 16
    SAVE_PATH = "best_model_multimodal.pth"  # 改个名字区分一下旧模型

    # 加载数据
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)

    if train_loader is not None and val_loader is not None:
        # 定义设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型：类别数改为 5
        num_classes = 5
        model = FruitClassifier(num_classes=num_classes).to(device)

        # 开始训练
        print("开始多模态融合模型训练...")
        train_model(model, train_loader, val_loader, EPOCHS, LEARNING_RATE, device, SAVE_PATH)
    else:
        print("数据加载失败，无法开始训练。请检查 data_loader.py")