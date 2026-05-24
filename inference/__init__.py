"""推理服务包。

对外暴露：
- InferencePreprocessor: 单样本预处理（与训练一致）
- ModelManager: ONNX 模型加载与推理
"""

from .preprocessing import InferencePreprocessor
from .model_loader import ModelManager

__all__ = ["InferencePreprocessor", "ModelManager"]
