import torch
from model import FruitClassifier
import os
import logging

# 配置日志
import torch
from model import FruitClassifier
import os
import logging
import onnx
import numpy as np

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def export_to_onnx(model_path, onnx_path, num_classes, input_size=(1, 3, 150, 150)):
    """
    导出模型为 ONNX 格式。

    Args:
        model_path (str): 已训练好的 PyTorch 模型路径 (.pth 文件)。
        onnx_path (str): 保存的 ONNX 文件路径。
        num_classes (int): 类别数量。
        input_size (tuple): 模型输入的张量形状 (默认 1x3x150x150)。
    """
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 初始化模型并加载权重
    model = FruitClassifier(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 创建一个假输入
    dummy_input = torch.randn(*input_size).to(device)

    # 导出模型
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,        # 保存权重参数
            opset_version=11,         # ONNX opset 版本
            do_constant_folding=True, # 常量折叠优化
            input_names=["input"],    # 输入名称
            output_names=["output"],  # 输出名称
            dynamic_axes={            # 支持动态输入形状
                "input": {0: "batch_size"},
                "output": {0: "batch_size"}
            }
        )
        logging.info(f"Model exported to {onnx_path} with input shape {input_size} and {num_classes} classes")

        # 验证导出的 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logging.info("ONNX model is valid.")

    except ImportError as e:
        logging.error(f"Failed to export model: ONNX module is not installed. Please install it with `pip install onnx`.")
    except Exception as e:
        logging.error(f"Failed to export model: {e}")


# 测试代码：仅供调试时使用
if __name__ == "__main__":
    MODEL_PATH = "final_model.pth"
    ONNX_PATH = "fruit_classifier.onnx"
    NUM_CLASSES = 2  # 类别数为2（fresh和rotten）

    export_to_onnx(MODEL_PATH, ONNX_PATH, NUM_CLASSES)