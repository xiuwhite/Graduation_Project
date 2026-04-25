import os
import torch
import random
from torchvision import datasets, transforms
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# 1. 修改为五级新鲜度映射
def map_labels(class_name):
    class_name = class_name.lower()
    if "very_fresh" in class_name:
        return 0
    elif "fresh" in class_name:  # 注意文件夹命名不要和very_fresh冲突，最好精确匹配
        return 1
    elif "slightly_old" in class_name:
        return 2
    elif "starting_rotten" in class_name:
        return 3
    elif "rotten" in class_name:
        return 4
    else:
        logging.error(f"Unknown class name: {class_name}")
        return -1


# 2. 核心创新点：模拟物理特征生成器
def simulate_physical_features(label):
    """
    根据新鲜度标签，生成模拟的 [重量, 体积, 硬度] 归一化特征 (0~1之间)。
    加入高斯噪声以模拟真实传感器的波动，防止模型死记硬背。
    """
    # 基础均值设定：[重量, 体积, 硬度]
    base_values = {
        0: [0.95, 0.95, 0.90],  # very_fresh: 水分足，体积饱满，硬度高
        1: [0.85, 0.85, 0.75],  # fresh: 轻微水分流失
        2: [0.70, 0.75, 0.50],  # slightly_old: 重量下降，开始变软
        3: [0.50, 0.60, 0.30],  # starting_rotten: 明显萎缩，局部软化
        4: [0.30, 0.40, 0.10],  # rotten: 严重脱水萎缩，结构破坏变软
    }

    base = base_values[label]
    # 添加标准差为0.05的随机噪声，并将数值截断在 [0, 1] 之间
    features = [max(0.0, min(1.0, random.gauss(mu, 0.05))) for mu in base]
    return torch.tensor(features, dtype=torch.float32)


# 3. 自定义数据集类：支持多模态输出
class FruitDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        # 重新映射标签
        self.targets = [map_labels(self.classes[label]) for label in self.targets]

        # 检查是否有无效标签
        invalid_labels = [i for i, target in enumerate(self.targets) if target == -1]
        if invalid_labels:
            logging.error(f"无效标签在图像索引: {invalid_labels}")

        # 更新 classes 列表为完整的5类
        self.classes = ["very_fresh", "fresh", "slightly_old", "starting_rotten", "rotten"]

    # 重写 __getitem__，让它每次吐出3个变量：图片、物理特征、标签
    def __getitem__(self, index):
        # 调用父类获取原始图片
        path, _ = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        # 获取映射后的真实标签
        target = self.targets[index]

        # 动态生成该样本对应的物理特征
        phys_feats = simulate_physical_features(target)

        return sample, phys_feats, target


def get_data_loaders(batch_size):
    # 定义数据变换 (保持你原来的数据增强逻辑)
    train_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    if not os.path.exists(DATA_DIR):
        logging.error(f"数据集路径不存在: {DATA_DIR}")
        return None, None

    # 加载数据
    try:
        train_dataset = FruitDataset(os.path.join(DATA_DIR, "train"), transform=train_transforms)
        validation_dataset = FruitDataset(os.path.join(DATA_DIR, "test"), transform=validation_transforms)
    except Exception as e:
        logging.error(f"加载数据集失败: {e}")
        return None, None

    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                                    pin_memory=True)

    logging.info("多模态数据加载器创建成功")
    return train_loader, validation_loader


if __name__ == "__main__":
    batch_size = 8
    train_loader, _ = get_data_loaders(batch_size)
    if train_loader:
        # 测试输出格式
        for images, phys_feats, labels in train_loader:
            print("图像形状:", images.shape)  # 应为 [8, 3, 150, 150]
            print("物理特征形状:", phys_feats.shape)  # 应为 [8, 3]
            print("标签形状:", labels.shape)  # 应为 [8]
            print("示例物理特征:", phys_feats[0])
            break