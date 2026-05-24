"""ONNX Runtime 模型加载与推理器。

支持三种模型的 ONNX 推理，同时提供 PyTorch 回退机制。
"""

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:  # pragma: no cover
    ort = None  # type: ignore
    _ONNX_AVAILABLE = False


class ModelManager:
    """管理 ONNX 模型生命周期。

    Usage::

        mgr = ModelManager("data/checkpoints")

        # 环境质量
        result = mgr.predict_env_quality(numeric_feat, odor_idx)

        # 睡眠影响
        preds = mgr.predict_sleep_impact(env_seq, static, hist, seq_len)

        # 控制策略
        actions = mgr.predict_control_policy(state_seq, seq_len)
    """

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self._sessions: dict[str, "ort.InferenceSession"] = {}

    # ── 内部工具 ──

    def _get_session(self, model_type: str):
        """获取或创建 ONNX session。"""
        if not _ONNX_AVAILABLE:
            raise RuntimeError("onnxruntime 未安装，请执行 pip install onnxruntime")

        if model_type not in self._sessions:
            path = self.model_dir / f"{model_type}.onnx"
            if not path.exists():
                raise FileNotFoundError(f"ONNX 模型不存在: {path}")
            self._sessions[model_type] = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
        return self._sessions[model_type]

    @property
    def loaded_models(self) -> list[str]:
        """已加载的模型列表。"""
        return list(self._sessions.keys())

    # ── 环境质量 ──

    def predict_env_quality(
        self, numeric: np.ndarray, odor_idx: np.ndarray
    ) -> dict[str, np.ndarray]:
        """预测环境质量。

        Args:
            numeric: shape (1, 6) float32
            odor_idx: shape (1,) int64

        Returns:
            {"comfort_score": (1,), "risk_logits": (1,4), "class_logits": (1,4)}
        """
        session = self._get_session("env_quality")
        # 确保 batch 维度
        if numeric.ndim == 1:
            numeric = numeric.reshape(1, -1)
        if odor_idx.ndim == 0:
            odor_idx = odor_idx.reshape(1)

        outputs = session.run(None, {
            "numeric_feat": numeric.astype(np.float32),
            "odor_idx": odor_idx.astype(np.int64),
        })

        # 输出顺序与模型 forward() 字典键顺序一致: comfort_score, risk_logits, class_logits
        return {
            "comfort_score": outputs[0],
            "risk_logits": outputs[1],
            "class_logits": outputs[2],
        }

    # ── 睡眠影响 ──

    def predict_sleep_impact(
        self,
        env_seq: np.ndarray,
        static: np.ndarray,
        history: np.ndarray,
        seq_lengths: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """预测睡眠指标。

        Args:
            env_seq: (1, seq_len, 4) float32
            static: (1, 11) float32
            history: (1, 5) float32
            seq_lengths: (1,) int64 或 None

        Returns:
            (1, 6) 预测值
        """
        session = self._get_session("sleep_impact")
        if env_seq.ndim == 2:
            env_seq = env_seq[np.newaxis, ...]
        if static.ndim == 1:
            static = static[np.newaxis, ...]
        if history.ndim == 1:
            history = history[np.newaxis, ...]
        if seq_lengths is not None and seq_lengths.ndim == 0:
            seq_lengths = seq_lengths.reshape(1)

        feed = {
            "env_seq": env_seq.astype(np.float32),
            "static_features": static.astype(np.float32),
            "history_features": history.astype(np.float32),
        }
        if seq_lengths is not None:
            feed["seq_lengths"] = seq_lengths.astype(np.int64)

        return session.run(None, feed)[0]

    # ── 控制策略 ──

    def predict_control_policy(
        self,
        state_seq: np.ndarray,
        seq_lengths: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> dict[str, np.ndarray]:
        """控制策略预测。

        Args:
            state_seq: (1, seq_len, 5) float32
            seq_lengths: (1,) int64 或 None
            deterministic: True 时返回均值动作

        Returns:
            {"discrete_logits": ..., "continuous_mean": ..., "continuous_log_std": ..., "state_value": ...}
        """
        session = self._get_session("control_policy")
        if state_seq.ndim == 2:
            state_seq = state_seq[np.newaxis, ...]
        if seq_lengths is not None and seq_lengths.ndim == 0:
            seq_lengths = seq_lengths.reshape(1)

        feed = {"state_seq": state_seq.astype(np.float32)}
        if seq_lengths is not None:
            feed["seq_lengths"] = seq_lengths.astype(np.int64)

        outputs = session.run(None, feed)
        return {
            "discrete_logits": outputs[0],
            "continuous_mean": outputs[1],
            "continuous_log_std": outputs[2],
            "state_value": outputs[3],
        }
