import torch
import torch.nn as nn
import torch.optim as optim
from model import FruitClassifier
from data_loader import get_data_loaders
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)

# 定义设备和超参数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
BATCH_SIZE = 64
SAVE_PATH = "best_model.pth"
FINAL_MODEL_PATH = "final_model.pth"
LOSS_CURVE_PATH = "loss_curve.png"

def train_and_validate():
    """
    训练并验证模型。
    """
    # 加载数据
    logging.info("加载数据集...")
    train_loader, val_loader = get_data_loaders(batch_size=BATCH_SIZE)

    if train_loader is None or val_loader is None:
        logging.error("数据加载失败，退出训练。")
        return

    # 初始化模型
    num_classes = 2  # 类别数为2（fresh和rotten）
    model = FruitClassifier(num_classes=num_classes).to(DEVICE)

    # 定义损失函数和优化器
    weights = torch.tensor([1.0, 2.0]).to(DEVICE)  # 可根据样本分布调整权重
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # 存储损失
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # 开始训练
    logging.info("开始训练模型...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # 验证模型
        val_loss, val_accuracy = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        logging.info(
            f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Train Loss: {train_losses[-1]:.4f}, "
            f"Validation Loss: {val_losses[-1]:.4f}, Accuracy: {val_accuracy:.2f}%"
        )

        # 保存最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            logging.info(f"模型在 Epoch {epoch + 1} 时保存，Validation Loss: {best_val_loss:.4f}")

        scheduler.step(val_loss)

    # 保存最终模型
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    logging.info(f"最终模型已保存为 {FINAL_MODEL_PATH}")

    # 绘制并保存损失曲线
    plot_loss_curve(train_losses, val_losses)
    logging.info(f"训练完成，最优模型已保存为 {SAVE_PATH}")

def validate(model, val_loader, criterion):
    """
    验证模型性能。

    Args:
        model (nn.Module): 模型
        val_loader (DataLoader): 验证数据加载器
        criterion (nn.Module): 损失函数

    Returns:
        tuple: 验证损失和准确率
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy

def plot_loss_curve(train_losses, val_losses):
    """
    绘制训练和验证损失曲线。

    Args:
        train_losses (list): 训练损失列表
        val_losses (list): 验证损失列表
    """
    plt.figure()
    plt.plot(range(NUM_EPOCHS), train_losses, label='Train Loss')
    plt.plot(range(NUM_EPOCHS), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.savefig(LOSS_CURVE_PATH)
    logging.info(f"损失曲线已保存到 {LOSS_CURVE_PATH}")
    plt.show()

if __name__ == "__main__":
    train_and_validate()
