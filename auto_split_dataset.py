import os
import random
import shutil

# ================= 配置区 =================
# 1. 改成你 E 盘里庞大的 train 文件夹路径
SOURCE_DIR = r"E:\dataset\dataset\train"

# 2. 改成你桌面的 train 文件夹路径
TARGET_DIR = r"C:\Users\20304\Desktop\fruit_freshness_project\data\train"

# 3. 训练集需要更多图片，设置为 300
IMAGES_PER_CLASS = 300
# ==========================================

TARGET_CLASSES = ["very_fresh", "fresh", "slightly_old", "starting_rotten", "rotten"]


def auto_split_dataset():
    # 1. 创建目标文件夹
    for cls in TARGET_CLASSES:
        os.makedirs(os.path.join(TARGET_DIR, cls), exist_ok=True)

    # 2. 收集原始图片路径
    fresh_images = []
    rotten_images = []

    print("正在扫描原始文件夹...")
    for folder_name in os.listdir(SOURCE_DIR):
        folder_path = os.path.join(SOURCE_DIR, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(folder_path, file)
                # 根据文件夹名字归类
                if "fresh" in folder_name.lower():
                    fresh_images.append(file_path)
                elif "rotten" in folder_name.lower():
                    rotten_images.append(file_path)

    print(f"扫描完毕！找到 Fresh 图片: {len(fresh_images)} 张，Rotten 图片: {len(rotten_images)} 张。")

    # 打乱顺序，保证随机性
    random.shuffle(fresh_images)
    random.shuffle(rotten_images)

    # 3. 检查数量是否足够
    required_fresh = int(IMAGES_PER_CLASS * 2.5)  # very_fresh, fresh, 加上一半的 slightly_old
    required_rotten = int(IMAGES_PER_CLASS * 2.5)  # rotten, starting_rotten, 加上一半的 slightly_old

    if len(fresh_images) < required_fresh or len(rotten_images) < required_rotten:
        print(f"❌ 警告：图片数量不足！你需要至少 {required_fresh} 张 fresh 和 {required_rotten} 张 rotten。")
        return

    print(f"\n开始随机抽取并复制图片 (每个类别 {IMAGES_PER_CLASS} 张)...")

    # 4. 智能分配逻辑
    # 类别 0: very_fresh (全部从 fresh 里拿)
    copy_files(fresh_images[:IMAGES_PER_CLASS], "very_fresh")
    del fresh_images[:IMAGES_PER_CLASS]

    # 类别 1: fresh (全部从剩下的 fresh 里拿)
    copy_files(fresh_images[:IMAGES_PER_CLASS], "fresh")
    del fresh_images[:IMAGES_PER_CLASS]

    # 类别 2: slightly_old (一半从 fresh 拿，一半从 rotten 拿，模拟过渡态)
    half = IMAGES_PER_CLASS // 2
    copy_files(fresh_images[:half], "slightly_old")
    del fresh_images[:half]
    copy_files(rotten_images[:(IMAGES_PER_CLASS - half)], "slightly_old")
    del rotten_images[:(IMAGES_PER_CLASS - half)]

    # 类别 3: starting_rotten (全部从 rotten 里拿)
    copy_files(rotten_images[:IMAGES_PER_CLASS], "starting_rotten")
    del rotten_images[:IMAGES_PER_CLASS]

    # 类别 4: rotten (全部从剩下的 rotten 里拿)
    copy_files(rotten_images[:IMAGES_PER_CLASS], "rotten")
    del rotten_images[:IMAGES_PER_CLASS]

    print("\n✅ 大功告成！图片已成功随机分配到目标文件夹。")


def copy_files(file_list, target_category):
    target_path = os.path.join(TARGET_DIR, target_category)
    for i, file in enumerate(file_list):
        # 重命名一下文件，防止不同文件夹里有同名文件冲突
        ext = os.path.splitext(file)[1]
        new_name = f"{target_category}_{i + 1:04d}{ext}"
        shutil.copy2(file, os.path.join(target_path, new_name))


if __name__ == "__main__":
    auto_split_dataset()