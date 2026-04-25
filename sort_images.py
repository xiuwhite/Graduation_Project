import os
import shutil
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

# ================= 配置区 =================
# 1. 你的“原始未分类图片”放在哪个文件夹？(请先把Kaggle下载的图片解压到这里)
SOURCE_DIR = r"raw_data"

# 2. 你分好类的图片要存到哪里？(直接存到你的数据集train目录下)
TARGET_BASE_DIR = r"data\train"

# 3. 5个目标类别的文件夹名 (必须和 data_loader.py 里一模一样)
CATEGORIES = {
    "1": "very_fresh",
    "2": "fresh",
    "3": "slightly_old",
    "4": "starting_rotten",
    "5": "rotten"
}


# ==========================================

class ImageSorterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("水果新鲜度快速分拣小工具 - 按 1~5 键分类，Space键跳过")
        self.root.geometry("600x650")

        # 确保目标文件夹存在
        for cat_folder in CATEGORIES.values():
            os.makedirs(os.path.join(TARGET_BASE_DIR, cat_folder), exist_ok=True)

        # 获取所有待分类的图片
        self.image_files = []
        if os.path.exists(SOURCE_DIR):
            for f in os.listdir(SOURCE_DIR):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_files.append(os.path.join(SOURCE_DIR, f))

        self.total_images = len(self.image_files)
        self.current_index = 0

        if self.total_images == 0:
            messagebox.showinfo("提示", f"在 {SOURCE_DIR} 中没有找到图片，请先放入图片！")
            self.root.destroy()
            return

        # UI 组件：进度条文本
        self.progress_label = tk.Label(root, text="", font=("Arial", 14))
        self.progress_label.pack(pady=10)

        # UI 组件：图片显示
        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        # UI 组件：操作提示
        instruction = "快捷键:\n[1] 非常新鲜  [2] 新鲜  [3] 轻微变质\n[4] 开始腐烂  [5] 烂透了\n[Space] 跳过这张  [Esc] 退出"
        self.info_label = tk.Label(root, text=instruction, font=("Arial", 12), fg="blue")
        self.info_label.pack(pady=10)

        # 绑定键盘事件
        self.root.bind("<Key>", self.handle_keypress)

        # 加载第一张图片
        self.load_image()

    def load_image(self):
        if self.current_index < self.total_images:
            img_path = self.image_files[self.current_index]
            self.progress_label.config(
                text=f"进度: {self.current_index + 1} / {self.total_images}\n当前文件: {os.path.basename(img_path)}")

            try:
                # 读取并缩放图片以适应窗口
                img = Image.open(img_path)
                img.thumbnail((400, 400))
                self.photo = ImageTk.PhotoImage(img)
                self.image_label.config(image=self.photo)
            except Exception as e:
                print(f"无法读取图片 {img_path}: {e}")
                self.next_image()
        else:
            messagebox.showinfo("完成", "恭喜你！所有图片都分拣完毕了！")
            self.root.destroy()

    def handle_keypress(self, event):
        key = event.char
        if key in CATEGORIES:
            # 移动文件
            src_path = self.image_files[self.current_index]
            target_folder = CATEGORIES[key]
            dest_path = os.path.join(TARGET_BASE_DIR, target_folder, os.path.basename(src_path))

            try:
                shutil.move(src_path, dest_path)
                print(f"[{target_folder}] <- {os.path.basename(src_path)}")
            except Exception as e:
                print(f"移动失败: {e}")

            self.next_image()
        elif event.keysym == 'space':
            print("跳过该图片")
            self.next_image()
        elif event.keysym == 'Escape':
            self.root.destroy()

    def next_image(self):
        self.current_index += 1
        self.load_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = ImageSorterApp(root)
    root.mainloop()