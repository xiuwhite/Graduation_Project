import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from tkinter import ttk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import os

from ttkthemes import ThemedTk
from model import FruitClassifier

# --- 1. 初始化模型与配置 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["非常新鲜 (Very Fresh)", "新鲜 (Fresh)", "轻微变质 (Slightly Old)", "开始腐烂 (Starting Rotten)",
               "完全腐烂 (Rotten)"]

model = FruitClassifier(num_classes=5).to(device)
MODEL_PATH = "best_model_multimodal.pth"
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
except Exception as e:
    messagebox.showerror("模型加载失败", f"找不到权重文件或加载失败:\n{e}")


def preprocess_image(image_path):
    try:
        preprocess = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])
        image = Image.open(image_path).convert("RGB")
        return preprocess(image).unsqueeze(0)
    except Exception as e:
        messagebox.showerror("错误", f"图片预处理失败: {e}")
        return None


def infer_multimodal(image_path, weight_val, volume_val, firmness_val):
    img_tensor = preprocess_image(image_path)
    if img_tensor is None: return None, 0

    phys_feats = torch.tensor([[weight_val, volume_val, firmness_val]], dtype=torch.float32)

    with torch.no_grad():
        outputs = model(img_tensor.to(device), phys_feats.to(device))
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_idx].item() * 100

    return CLASS_NAMES[predicted_idx], confidence


# --- 2. 现代化 GUI 界面构建 ---
root = ThemedTk(theme="arc")
root.title("多模态水果新鲜度智能检测系统 V2.0")
root.geometry("880x720")
root.configure(bg="#f5f6f7")

# ================= 顶部：标题栏 =================
header_frame = tk.Frame(root, bg="#2c3e50", height=70)
header_frame.pack(fill=tk.X, side=tk.TOP)
header_frame.pack_propagate(False)
tk.Label(header_frame, text="多特征融合：水果新鲜度智能评估系统",
         font=("Microsoft YaHei", 18, "bold"), fg="white", bg="#2c3e50").pack(pady=15)

main_frame = tk.Frame(root, bg="#f5f6f7")
main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=15)

# ----------------- 左侧：视觉输入区 -----------------
left_frame = ttk.LabelFrame(main_frame, text=" 📷 步骤 1：视觉图像采集 ")
left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=5, ipadx=10, ipady=10)

# 给图片加一个固定大小的容器，绝对不会变形和消失
img_container = tk.Frame(left_frame, width=260, height=260, bg="#e1e8ed")
img_container.pack(pady=20, padx=20)
img_container.pack_propagate(False)  # 锁定大小

image_label = tk.Label(img_container, text="点击下方按钮\n上传水果图片",
                       font=("Microsoft YaHei", 12), bg="#e1e8ed", fg="#7f8c8d")
image_label.pack(expand=True, fill=tk.BOTH)

current_image_path = None


def upload_image():
    global current_image_path
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        current_image_path = file_path
        # 强制转为 RGB 并用更稳妥的 resize 方法
        img = Image.open(file_path).convert("RGB")
        img = img.resize((260, 260), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        image_label.config(image=photo, text="")
        image_label.image = photo

        result_text.config(state=tk.NORMAL)
        result_text.insert(tk.END, f"\n[System] 📷 成功加载图像: {os.path.basename(file_path)}\n")
        result_text.see(tk.END)
        result_text.config(state=tk.DISABLED)


style = ttk.Style()
style.configure("TButton", font=("Microsoft YaHei", 11))

upload_btn = ttk.Button(left_frame, text="📂 浏览并上传图像", command=upload_image)
upload_btn.pack(pady=10, ipadx=15, ipady=5)

# ----------------- 右侧：传感器与终端区 -----------------
right_frame = tk.Frame(main_frame, bg="#f5f6f7")
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=5)

sensor_frame = ttk.LabelFrame(right_frame, text=" 🎛️ 步骤 2：模拟物理传感器参数 ")
sensor_frame.pack(fill=tk.X, pady=(0, 10), ipadx=10, ipady=10)


def create_slider(parent, label_text, default_val):
    frame = tk.Frame(parent, bg="#ffffff")
    frame.pack(fill=tk.X, pady=8, padx=15)

    tk.Label(frame, text=label_text, font=("Microsoft YaHei", 10, "bold"),
             width=16, anchor="w", bg="#ffffff", fg="#34495e").pack(side=tk.LEFT)

    var = tk.DoubleVar(value=default_val)
    slider = ttk.Scale(frame, from_=0.0, to=1.0, variable=var, orient=tk.HORIZONTAL)
    slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=15)

    val_label = tk.Label(frame, text=f"{default_val:.2f}", font=("Consolas", 11, "bold"),
                         width=5, bg="#e8f4f8", fg="#2980b9")
    val_label.pack(side=tk.LEFT)

    def update_label(*args):
        val_label.config(text=f"{var.get():.2f}")

    var.trace_add("write", update_label)
    return var


slider_container = tk.Frame(sensor_frame, bg="#ffffff", bd=1, relief="solid")
slider_container.pack(fill=tk.X, padx=10, pady=10)

weight_var = create_slider(slider_container, "⚖️ 重量留存率 (0-1)", 0.85)
volume_var = create_slider(slider_container, "🍎 体积饱满度 (0-1)", 0.85)
firmness_var = create_slider(slider_container, "🔨 表面硬度 (0-1)", 0.75)

tk.Label(sensor_frame, text="💡 提示：越接近 1.0 表示物理状态越完好，越接近 0.0 表示越萎缩软烂",
         font=("Microsoft YaHei", 9), fg="#7f8c8d", bg="#f5f6f7").pack(anchor=tk.W, padx=10)


def run_detection():
    if not current_image_path:
        messagebox.showwarning("操作错误", "请先在左侧上传一张水果图像！")
        return

    w_val, v_val, f_val = weight_var.get(), volume_var.get(), firmness_var.get()
    pred_class, conf = infer_multimodal(current_image_path, w_val, v_val, f_val)

    result_text.config(state=tk.NORMAL)
    result_text.insert(tk.END, "-" * 58 + "\n")
    result_text.insert(tk.END, f"🚀 发起多模态联合评估...\n")
    result_text.insert(tk.END, f" ├─ 视觉特征输入: {os.path.basename(current_image_path)}\n")
    result_text.insert(tk.END, f" └─ 物理特征输入: [重量:{w_val:.2f}, 体积:{v_val:.2f}, 硬度:{f_val:.2f}]\n\n")

    if pred_class:
        result_text.insert(tk.END, f"✅ 综合诊断结果: 【 {pred_class} 】\n")
        result_text.insert(tk.END, f"📊 系统置信度:   {conf:.2f}%\n")

    result_text.see(tk.END)
    result_text.config(state=tk.DISABLED)


detect_btn = tk.Button(right_frame, text="⚡ 开始融合评估 ⚡", font=("Microsoft YaHei", 14, "bold"),
                       bg="#27ae60", fg="white", activebackground="#2ecc71", activeforeground="white",
                       relief="flat", cursor="hand2", command=run_detection)
detect_btn.pack(fill=tk.X, pady=(5, 15), ipady=8)

terminal_frame = ttk.LabelFrame(right_frame, text=" 📋 智能诊断终端输出 ")
terminal_frame.pack(fill=tk.BOTH, expand=True, ipadx=5, ipady=5)

# 更换终端配色：更柔和的深灰背景 + 高亮荧光绿 + 大号加粗字体
result_text = ScrolledText(terminal_frame, font=("Consolas", 11, "bold"),
                           bg="#282C34", fg="#00FA9A",  # One Dark背景色 + 春绿色
                           bd=0, padx=10, pady=10, insertbackground="white")
result_text.pack(fill=tk.BOTH, expand=True)

result_text.insert(tk.END, ">>> 多特征融合水果新鲜度评估系统已启动。\n")
result_text.insert(tk.END, ">>> 正在等待用户载入视觉图像与物理参数...\n")
result_text.config(state=tk.DISABLED)

root.mainloop()  # <--- 就是这一句，程序能不能跑起来全靠它！