import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class FruitClassifier(nn.Module):
    """
    基于多特征融合（视觉+物理）的水果新鲜度五级分类模型。
    """

    def __init__(self, num_classes=5, dropout_rate=0.5):
        super(FruitClassifier, self).__init__()

        # ==========================================
        # 1. 视觉特征提取模块 (保留你原本的CNN结构)
        # ==========================================
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)  # 输出: (150x150)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样到 (75x75)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # 输出: (75x75)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样到 (37x37)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 输出: (37x37)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # 下采样到 (18x18)

        # 视觉特征映射层
        self.dropout_rate = dropout_rate
        self.fc_vis = nn.Linear(128 * 18 * 18, 512)
        self.dropout = nn.Dropout(dropout_rate)

        # ==========================================
        # 2. 物理特征处理模块 (新增：毕设创新点)
        # ==========================================
        # 物理特征输入维度为3 (重量, 体积, 硬度)，我们把它映射到 16 维空间
        self.fc_phys = nn.Linear(3, 16)

        # ==========================================
        # 3. 特征融合与分类模块 (新增)
        # ==========================================
        # 融合后的总维度 = 视觉特征(512) + 物理特征(16) = 528
        self.fc_final = nn.Linear(512 + 16, num_classes)

    def forward(self, images, phys_feats):
        """
        前向传播函数，支持双模态输入。
        Args:
            images: 图像张量 [batch_size, 3, 150, 150]
            phys_feats: 物理特征张量 [batch_size, 3]
        """
        # --- 1. 提取视觉特征 ---
        x_vis = self.pool1(F.relu(self.bn1(self.conv1(images))))
        x_vis = self.pool2(F.relu(self.bn2(self.conv2(x_vis))))
        x_vis = self.pool3(F.relu(self.bn3(self.conv3(x_vis))))

        x_vis = x_vis.view(x_vis.size(0), -1)  # 展平
        x_vis = F.relu(self.fc_vis(x_vis))
        x_vis = self.dropout(x_vis)  # 此时视觉特征形状: [batch_size, 512]

        # --- 2. 提取物理特征 ---
        x_phys = F.relu(self.fc_phys(phys_feats))  # 此时物理特征形状: [batch_size, 16]

        # --- 3. 多模态特征拼接 (Concat) ---
        x_fused = torch.cat((x_vis, x_phys), dim=1)  # 融合后形状: [batch_size, 528]

        # --- 4. 最终输出分类结果 ---
        out = self.fc_final(x_fused)
        return out


# 测试代码
if __name__ == "__main__":
    # 初始化五分类模型
    model = FruitClassifier(num_classes=5).to(device)

    # 模拟 data_loader 吐出的数据
    sample_images = torch.randn(8, 3, 150, 150).to(device)  # 模拟8张图片
    sample_phys = torch.randn(8, 3).to(device)  # 模拟8组物理特征

    # 测试模型前向传播
    sample_output = model(sample_images, sample_phys)

    print("\n--- 融合模型测试成功 ---")
    print("输入图像形状:", sample_images.shape)
    print("输入物理特征形状:", sample_phys.shape)
    print("模型输出形状:", sample_output.shape)  # 应该是 [8, 5]