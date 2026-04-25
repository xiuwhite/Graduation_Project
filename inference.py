import onnxruntime as ort
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

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
    # 定义预处理步骤
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 加载图像并应用预处理
    try:
        image = Image.open(image_path).convert("RGB")
        image = preprocess(image)
        return image.unsqueeze(0).numpy()  # 添加批次维度并转换为 NumPy 数组
    except Exception as e:
        logging.error(f"Failed to preprocess image: {e}")
        return None


def infer_onnx_model(onnx_path, image_path, class_names):
    """
    使用 ONNX 模型进行推理。

    Args:
        onnx_path (str): ONNX 模型文件路径。
        image_path (str): 输入图像文件路径。
        class_names (list): 类别名称列表。

    Returns:
        str: 推理结果类别名称。
    """
    # 检查 ONNX 模型文件是否存在
    if not os.path.exists(onnx_path):
        logging.error(f"ONNX model file not found: {onnx_path}")
        return None

    # 检查输入图像文件是否存在
    if not os.path.exists(image_path):
        logging.error(f"Image file not found: {image_path}")
        return None

    # 加载 ONNX 模型
    try:
        ort_session = ort.InferenceSession(onnx_path)
    except Exception as e:
        logging.error(f"Failed to load ONNX model: {e}")
        return None

    # 预处理输入图像
    input_tensor = preprocess_image(image_path)
    if input_tensor is None:
        return None

    # 运行推理
    try:
        ort_inputs = {"input": input_tensor}
        ort_outputs = ort_session.run(None, ort_inputs)
        logits = ort_outputs[0]  # 获取 logits
        logging.info(f"模型原始输出 (logits): {logits}")  # 打印 logits

        # 计算 Softmax 概率
        probabilities = softmax(logits)
        logging.info(f"Softmax 概率: {probabilities}")  # 打印概率分布

        # 打印每个类别的概率
        for i, class_name in enumerate(class_names):
            logging.info(f"{class_name} 的概率: {probabilities[0][i]:.6f}")
    except Exception as e:
        logging.error(f"Failed to run inference: {e}")
        return None

    # 获取预测类别
    predicted_class = np.argmax(probabilities)  # 基于概率分布选择类别
    logging.info(f"预测类别索引: {predicted_class}")  # 打印预测类别索引
    return class_names[predicted_class]


# 测试代码：支持批量测试图片
if __name__ == "__main__":
    ONNX_PATH = "fruit_classifier.onnx"
    CLASS_NAMES = ["Fresh", "Rotten"]  # 类别数为2（Fresh和Rotten）

    # 测试图片路径列表
    IMAGE_PATHS = [
        "test_images/fresh_fruit.png",  # 替换为你的测试图片路径
        "test_images/rotten_fruit.png",  # 替换为你的测试图片路径

        # 添加更多图片路径...
    ]

    # 遍历测试图片并推理
    for image_path in IMAGE_PATHS:
        if not os.path.exists(image_path):
            logging.error(f"图片文件不存在: {image_path}")
            continue

        # 推理图像
        result = infer_onnx_model(ONNX_PATH, image_path, CLASS_NAMES)
        if result is not None:
            logging.info(f"图片: {image_path}\n预测结果: {result}\n")


