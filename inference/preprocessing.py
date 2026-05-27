"""单样本预处理管线。

严格遵循 [`train.py`](../train.py) 中 [`SleepDataset._preprocess`](../train.py:93) 的逻辑，
确保推理时的数据变换与训练时完全一致。

所有列名从 [`DataConfig`](config.py) 导入，避免重复定义。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from config import DataConfig, get_default_data_config


class InferencePreprocessor:
    """推理预处理管线。

    从训练阶段保存的 `preprocessing.pkl` 加载所有 Scaler，
    对外提供三个模型的预处理接口。

    Usage::

        preprocessor = InferencePreprocessor("data/checkpoints/preprocessing.pkl")

        # 环境质量
        numeric, odor_idx = preprocessor.env_quality(raw_env_dict)

        # 睡眠影响
        env_seq, static, hist, seq_len = preprocessor.sleep_impact(raw_sleep_dict)

        # 控制策略
        state_seq, seq_len = preprocessor.control_policy(raw_ctrl_dict)
    """

    def __init__(self, artifact_path: str, data_config: Optional[DataConfig] = None):
        artifacts = joblib.load(artifact_path)

        self.env_scaler: StandardScaler = artifacts.get("env_scaler")
        self.static_scaler: StandardScaler = artifacts.get("static_scaler")
        self.sleep_scaler: StandardScaler = artifacts.get("sleep_scaler")
        self.control_scaler: StandardScaler = artifacts.get("control_scaler")

        # 验证关键 Scaler 存在
        for name in ("env_scaler", "static_scaler", "sleep_scaler", "control_scaler"):
            if getattr(self, name) is None:
                raise ValueError(f"缺少 Scaler: {name}，请确认 preprocessing.pkl 是否正确")

        # 列名配置
        self.cfg = data_config or get_default_data_config()

    # ──────────────────── 环境质量 ────────────────────

    def env_quality(self, raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """处理单条环境数据 → (numeric (6,), odor_idx (scalar)).

        raw 应包含字段与 ``DataConfig.env_numeric_cols`` 一致。
        """
        # 1. 构建数值特征向量
        numeric = np.array(
            [raw.get(c, 0.0) for c in self.cfg.env_numeric_cols], dtype=np.float32
        )
        numeric = self.env_scaler.transform(numeric.reshape(1, -1)).reshape(-1).astype(np.float32)

        # 2. 气味编码
        odor_str = raw.get("odor_type", "无")
        odor_idx = np.array(self.cfg.odor_type_mapping.get(odor_str, 0), dtype=np.int64)

        return numeric, odor_idx

    # ──────────────────── 睡眠影响 ────────────────────

    def sleep_impact(
        self,
        env_history: list[dict],
        static_raw: dict,
        prev_sleep: Optional[dict] = None,
        seq_len: int = 24,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """处理单用户睡眠数据 → (env_seq, static, hist, seq_len_tensor).

        Args:
            env_history: 最近 N 条环境数据
            static_raw: 静态协变量字典
            prev_sleep: 前一日睡眠指标（可选，无则填零）
            seq_len: 最大序列长度（需与训练时一致）

        Returns:
            env_seq: (seq_len, env_seq_dim) 标准化环境序列
            static: (static_dim,) 静态特征
            hist: (hist_dim,) 历史睡眠指标
            seq_len_tensor: (1,) 有效时间步数
        """
        env_seq_cols = self.cfg.env_seq_cols
        static_cols = self.cfg.static_cols
        static_scale_cols = self.cfg.static_scale_cols
        sleep_base_cols = self.cfg.sleep_base_cols

        # 1. 环境序列：取最后 seq_len 条
        env_list = env_history[-seq_len:]
        raw_seq_len = len(env_list)
        env_arr = np.zeros((raw_seq_len, len(env_seq_cols)), dtype=np.float32)
        for t, row in enumerate(env_list):
            env_arr[t] = [row.get(c, 0.0) for c in env_seq_cols]

        # 标准化后用零填充到 seq_len
        # 注意：env_scaler 拟合在 env_numeric_cols（6 列）上，而 env_seq 只用前 4 列。
        # 训练时 env_seq 取自已标准化的 DataFrame，故无需调用 transform()；
        # 推理时手动使用前 env_seq_dim 列的 mean_/scale_ 做标准化。
        n_seq_cols = len(env_seq_cols)
        if raw_seq_len > 0:
            env_arr = (
                (env_arr - self.env_scaler.mean_[:n_seq_cols]) / self.env_scaler.scale_[:n_seq_cols]
            ).astype(np.float32)

        env_seq = np.zeros((seq_len, n_seq_cols), dtype=np.float32)
        if raw_seq_len >= seq_len:
            env_seq = env_arr[:seq_len].astype(np.float32)
        elif raw_seq_len > 0:
            env_seq[seq_len - raw_seq_len:] = env_arr.astype(np.float32)

        # 2. 静态特征
        static_vals = []
        for c in static_cols:
            val = static_raw.get(c, 0)
            static_vals.append(float(val) if not isinstance(val, (int, float)) else val)
        static_np = np.array(static_vals, dtype=np.float32)

        # 仅对数值列做标准化
        scale_cols_list = list(static_scale_cols)
        for idx, col in enumerate(static_cols):
            if col in scale_cols_list:
                col_idx = scale_cols_list.index(col)
                static_np[idx] = (
                    static_np[idx] - self.static_scaler.mean_[col_idx]
                ) / self.static_scaler.scale_[col_idx]

        # 3. 历史睡眠指标
        if prev_sleep and any(prev_sleep.get(c) is not None for c in sleep_base_cols):
            hist_arr = np.array(
                [prev_sleep.get(c, 0.0) for c in sleep_base_cols], dtype=np.float32
            )
            hist_arr = self.sleep_scaler.transform(hist_arr.reshape(1, -1)).reshape(-1).astype(np.float32)
        else:
            hist_arr = np.zeros(len(sleep_base_cols), dtype=np.float32)

        seq_len_tensor = np.array(max(1, min(raw_seq_len, seq_len)), dtype=np.int64)

        return env_seq, static_np, hist_arr, seq_len_tensor

    # ──────────────────── 控制策略 ────────────────────

    def control_policy(
        self,
        state_history: list[dict],
        seq_len: int = 24,
    ) -> tuple[np.ndarray, np.ndarray]:
        """处理控制策略状态序列 → (state_seq, seq_len_tensor).

        Args:
            state_history: 最近 N 条状态数据
            seq_len: 最大序列长度（需与训练时一致）

        Returns:
            state_seq: (seq_len, state_dim)
            seq_len_tensor: (1,)
        """
        ctrl_state_cols = self.cfg.control_state_cols
        state_list = state_history[-seq_len:]
        raw_seq_len = len(state_list)
        state_arr = np.zeros((raw_seq_len, len(ctrl_state_cols)), dtype=np.float32)
        for t, row in enumerate(state_list):
            state_arr[t] = [row.get(c, 0.0) for c in ctrl_state_cols]

        if raw_seq_len > 0:
            state_arr = self.control_scaler.transform(state_arr).astype(np.float32)

        state_seq = np.zeros((seq_len, len(ctrl_state_cols)), dtype=np.float32)
        if raw_seq_len >= seq_len:
            state_seq = state_arr[:seq_len].astype(np.float32)
        elif raw_seq_len > 0:
            state_seq[seq_len - raw_seq_len:] = state_arr.astype(np.float32)

        seq_len_tensor = np.array(max(1, min(raw_seq_len, seq_len)), dtype=np.int64)
        return state_seq, seq_len_tensor
