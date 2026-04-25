import torch
from model import FruitClassifier
from data_loader import get_data_loaders
from evaluate import evaluate_model
import os
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 数据集与模型路径
TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
MODEL_PATH = "best_model_multimodal.pth"  # 改为咱们训练好的多模态权重名
CLASS_NAMES = ["very_fresh", "fresh", "slightly_old", "starting_rotten", "rotten"]  # 升级为五分类


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. 数据加载测试 (支持多模态输出)
    logging.info("加载多模态数据...")
    try:
        train_loader, validation_loader = get_data_loaders(batch_size=32)
        # 遍历训练数据示例：注意这里解包出了 phys_feats
        for batch_idx, (images, phys_feats, labels) in enumerate(train_loader):
            logging.info(f"Batch {batch_idx + 1}:")
            logging.info(f"Image batch shape: {images.shape}")
            logging.info(f"Phys features shape: {phys_feats.shape}")
            logging.info(f"Label batch shape: {labels.shape}")
            break
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        return

    # 2. 模型定义与测试 (支持双分支输入)
    logging.info("定义和测试多特征融合模型...")
    try:
        num_classes = 5  # 类别数改为5
        model = FruitClassifier(num_classes=num_classes).to(device)

        # 测试模型：同时生成模拟图像和模拟物理特征
        sample_input_img = torch.randn(16, 3, 150, 150).to(device)  # Batch size = 16
        sample_input_phys = torch.randn(16, 3).to(device)  # 物理特征 (重量, 体积, 硬度)

        output = model(sample_input_img, sample_input_phys)
        logging.info(f"模型输出形状: {output.shape}")  # 应为 [16, 5]
    except Exception as e:
        logging.error(f"模型定义与测试失败: {e}")
        return

    # 3. 加载已训练好的模型并评估
    logging.info("加载已训练多模态模型并进行测试集评估...")
    try:
        if not os.path.exists(MODEL_PATH):
            logging.error(f"模型文件不存在: {MODEL_PATH}，请先运行 train.py 进行训练！")
            return

        # 添加 weights_only=True 解决安全警告
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        _, test_loader = get_data_loaders(batch_size=32)

        # 调用评估模块
        evaluate_model(model, test_loader, CLASS_NAMES, device)

    except Exception as e:
        logging.error(f"模型加载与评估失败: {e}")
        return

    logging.info("系统串联测试任务完成！")


if __name__ == "__main__":
    main()