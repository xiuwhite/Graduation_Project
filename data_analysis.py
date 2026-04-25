import os
from collections import Counter
from torchvision import datasets
import matplotlib.pyplot as plt
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# 数据集路径
DATA_DIR = os.path.abspath("C:/Users/20304/PycharmProjects/pythonProject/data/")


def analyze_data(dataset_path):
    """
    分析数据集，包括类别分布和样本数量。
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

    dataset = datasets.ImageFolder(dataset_path)
    class_names = dataset.classes  # 获取类别名称
    class_counts = Counter()

    for _, label in dataset.samples:
        class_name = class_names[label]
        class_counts[class_name] += 1

    logging.info(f"类别分布: {class_counts}")
    return class_counts, class_names


def visualize_class_distribution(class_counts, class_names):
    """
    可视化类别分布。
    """
    plt.figure(figsize=(10, 6))
    plt.bar(class_names, [class_counts.get(cls, 0) for cls in class_names], color='skyblue')
    plt.xlabel("Class")
    plt.ylabel("Sample Count")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.show()


def visualize_sample_images(dataset_path, num_samples=5):
    """
    可视化每个类别的样本图像。
    """
    dataset = datasets.ImageFolder(dataset_path)
    class_names = dataset.classes

    # 收集每个类别的样本
    class_samples = {cls: [] for cls in class_names}
    for img_path, label in dataset.samples:
        class_name = class_names[label]
        if len(class_samples[class_name]) < num_samples:
            class_samples[class_name].append(img_path)

    # 可视化样本
    plt.figure(figsize=(15, 4))
    for class_idx, (class_name, images) in enumerate(class_samples.items()):
        if not images:
            continue
        for i, img_path in enumerate(images):
            img = plt.imread(img_path)
            plt.subplot(len(class_samples), num_samples, class_idx * num_samples + i + 1)
            plt.imshow(img)
            plt.title(f"{class_name} ({i + 1})")
            plt.axis('off')
    plt.tight_layout()
    plt.show()


# 测试代码：仅供调试时使用
if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, "train")
    class_counts, class_names = analyze_data(train_path)
    visualize_class_distribution(class_counts, class_names)
    visualize_sample_images(train_path, num_samples=5)
