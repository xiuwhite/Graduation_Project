import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

def softmax(x):
    """计算 Softmax"""
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # 防止数值溢出
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def preprocess_image(image_path, input_size=(150, 150)):
    """
    对输入图像进行预处理。

    Args:
        image_path (str): 图像文件路径。
        input_size (tuple): 模型输入图像的尺寸 (宽, 高)。

    Returns:
        numpy.ndarray: 预处理后的图像张量。
    """
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image)
    return image.unsqueeze(0).numpy()  # 添加批次维度

# 加载 ONNX 模型
onnx_path = "fruit_classifier.onnx"  # 替换为你的模型路径
session = ort.InferenceSession(onnx_path)

# 获取输入信息
input_name = session.get_inputs()[0].name
print("Model input name:", input_name)

# 加载真实图像
image_path = "test_images/rotten_fruit.png"  # 替换为测试图像路径
dummy_input = preprocess_image(image_path)
print("Input shape:", dummy_input.shape)

# 运行推理
output = session.run(None, {input_name: dummy_input})
logits = output[0]  # 模型原始输出
print("Logits:", logits)

# 计算 Softmax 概率
probabilities = softmax(logits)
print("Probabilities:", probabilities)

# 解释结果
labels = ["fresh", "rotten"]  # 类别名称
predicted_class = labels[np.argmax(probabilities)]
print(f"Predicted class: {predicted_class}")
