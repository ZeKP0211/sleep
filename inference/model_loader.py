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
        self._preload()

    # ── 预加载 ──

    def _preload(self) -> None:
        """启动时预加载目录中所有 ONNX 模型，使健康检查立即可用。"""
        if not _ONNX_AVAILABLE:
            return
        for path in sorted(self.model_dir.glob("*.onnx")):
            model_type = path.stem
            try:
                self._sessions[model_type] = ort.InferenceSession(
                    str(path), providers=["CPUExecutionProvider"]
                )
            except Exception:
                pass

    # ── 内部工具 ──

    def _get_session(self, model_type: str):
        """获取或创建 ONNX session（未预加载时按需加载）。"""
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

    # ── 静态工具：从分布参数计算动作 ──

    @staticmethod
    def _sample_discrete(logits: np.ndarray) -> np.ndarray:
        """从 logits 采样离散动作：Categorical.sample()。"""
        probs = _softmax(logits, axis=-1)
        cumsum = np.cumsum(probs, axis=-1)
        r = np.random.uniform(size=probs.shape[:-1] + (1,))
        return np.argmax(r < cumsum, axis=-1).astype(np.int64)

    @staticmethod
    def _sample_continuous(mean: np.ndarray, log_std: np.ndarray) -> np.ndarray:
        """从 Normal(mean, exp(log_std)) 采样连续动作。"""
        std = np.exp(log_std)
        return mean + std * np.random.randn(*mean.shape).astype(np.float32)

    @staticmethod
    def _deterministic_continuous(mean: np.ndarray) -> np.ndarray:
        """确定性连续动作 = mean。"""
        return mean.astype(np.float32)

    # ── 环境质量 ──

    def predict_env_quality(
        self, numeric: np.ndarray, odor_idx: np.ndarray
    ) -> dict[str, np.ndarray]:
        """预测环境质量。

        Args:
            numeric: shape (1, numeric_dim) float32，维度由导出时模型配置决定
            odor_idx: shape (1,) int64

        Returns:
            {"comfort_score": (1,), "risk_logits": (1,risk_dim), "class_logits": (1,num_classes)}
            各维度由导出时模型配置决定
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
            env_seq: (1, seq_len, env_seq_dim) float32，维度由导出时模型配置决定
            static: (1, static_dim) float32
            history: (1, hist_dim) float32
            seq_lengths: (1,) int64 或 None

        Returns:
            (1, output_dim) 预测值，维度由导出时模型配置决定
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
        """控制策略预测：从 ONNX forward 输出计算动作。

        模型导出时仅包含 forward()（logits / mean / log_std / value），
        采样逻辑在此处实现，与 PyTorch 版本完全对齐。

        Args:
            state_seq: (1, seq_len, state_dim) float32
            seq_lengths: (1,) int64 或 None
            deterministic: True → argmax 离散 + mean 连续
                          False → Categorical.sample() + Normal.sample()

        Returns:
            {
                "discrete_action": (1,) int64,
                "continuous_action": (1, cont_dim) float32,
                "state_value": (1,) float32,
                "discrete_logits": (1, disc_dim) float32,
                "continuous_mean": (1, cont_dim) float32,
                "continuous_log_std": (1, cont_dim) float32,
            }
        """
        session = self._get_session("control_policy")
        if state_seq.ndim == 2:
            state_seq = state_seq[np.newaxis, ...]
        if seq_lengths is not None and seq_lengths.ndim == 0:
            seq_lengths = seq_lengths.reshape(1)

        seq_len = state_seq.shape[1] if state_seq.ndim == 3 else state_seq.shape[0]
        if seq_lengths is None:
            seq_lengths = np.array([seq_len], dtype=np.int64)
        elif seq_lengths.ndim == 0:
            seq_lengths = seq_lengths.reshape(1)

        feed = {
            "state_seq": state_seq.astype(np.float32),
            "seq_lengths": seq_lengths.astype(np.int64),
        }

        outputs = session.run(None, feed)
        discrete_logits = outputs[0]       # (1, discrete_action_dim)
        continuous_mean = outputs[1]       # (1, continuous_action_dim)
        continuous_log_std = outputs[2]    # (1, continuous_action_dim)
        state_value = outputs[3]           # (1,)

        # ── 动作选择 ──
        if deterministic:
            discrete_action = np.argmax(discrete_logits, axis=-1).astype(np.int64)
            continuous_action = self._deterministic_continuous(continuous_mean)
        else:
            discrete_action = self._sample_discrete(discrete_logits)
            continuous_action = self._sample_continuous(continuous_mean, continuous_log_std)

        return {
            "discrete_action": discrete_action,
            "continuous_action": continuous_action,
            "state_value": state_value,
            "discrete_logits": discrete_logits,
            "continuous_mean": continuous_mean,
            "continuous_log_std": continuous_log_std,
        }


# ── 独立工具函数 ──

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """稳定 softmax 实现。"""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)