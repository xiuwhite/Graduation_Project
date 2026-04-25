import torch
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, class_names, device):
    """
    在测试集上评估多模态模型性能。
    """
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        # 注意这里：解包出 3 个变量 (加上了物理特征)
        for images, phys_feats, labels in tqdm(test_loader, desc="Evaluating", leave=False):
            images = images.to(device)
            phys_feats = phys_feats.to(device)
            labels = labels.to(device)

            # 前向传播：同时传入图像和物理特征
            outputs = model(images, phys_feats)
            _, predicted = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())

    # 生成分类报告
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))

    # 绘制混淆矩阵 (毕设论文凑字数/上档次必备图表)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Multi-modal Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    from model import FruitClassifier
    from data_loader import get_data_loaders

    # 类别名称更新为5类
    CLASS_NAMES = ["very_fresh", "fresh", "slightly_old", "starting_rotten", "rotten"]

    # 加载测试数据 (这里只需测试集即可，batch_size稍微大点没关系)
    _, test_loader = get_data_loaders(batch_size=32)

    if test_loader is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化模型 (5分类)
        num_classes = len(CLASS_NAMES)
        model = FruitClassifier(num_classes=num_classes).to(device)

        # 加载刚才训练好的多模态模型权重
        MODEL_PATH = "best_model_multimodal.pth"
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print(f"成功加载模型权重: {MODEL_PATH}")
            # 评估模型
            evaluate_model(model, test_loader, CLASS_NAMES, device)
        except Exception as e:
            print(f"加载模型失败: {e}")