"""单样本预处理管线。

严格遵循 [`train.py`](../train.py) 中 [`SleepDataset._preprocess`](../train.py:93) 的逻辑，
确保推理时的数据变换与训练时完全一致。
"""

from pathlib import Path
from typing import Optional

import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler


# ── 气味映射（与 train.py 一致） ──
_ODOR_TYPE_MAPPING = {"无": 0, "薰衣草": 1, "沉香": 2, "川芎": 3, "其他": 4}

# ── 列名常量（与 train.py 一致） ──
_ENV_NUMERIC_COLS = ["temp", "humidity", "temp_humidity_interaction", "odor_intensity", "odor_duration", "odor_preference"]
_ENV_SEQ_COLS = ["temp", "humidity", "temp_humidity_interaction", "odor_intensity"]
_STATIC_COLS = ["age", "gender", "bmi", "season", "health_nose", "health_asthma", "health_depression",
                "habit_alcohol", "habit_caffeine", "habit_exercise", "habit_screen_time"]
# static_scaler 仅对以下数值列做标准化（其余为 binary/categorical，不做变换）
_STATIC_SCALE_COLS = ["age", "bmi", "season", "habit_alcohol", "habit_caffeine", "habit_exercise", "habit_screen_time"]
_SLEEP_BASE_COLS = ["sleep_efficiency", "sleep_latency", "deep_sleep_duration", "awakenings", "apnea_index"]
_CONTROL_STATE_COLS = ["temp", "humidity", "odor_intensity", "sleep_stage", "time_of_day"]


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

    def __init__(self, artifact_path: str):
        artifacts = joblib.load(artifact_path)

        self.env_scaler: StandardScaler = artifacts.get("env_scaler")
        self.static_scaler: StandardScaler = artifacts.get("static_scaler")
        self.sleep_scaler: StandardScaler = artifacts.get("sleep_scaler")
        self.control_scaler: StandardScaler = artifacts.get("control_scaler")

        # 验证关键 Scaler 存在
        for name in ("env_scaler", "static_scaler", "sleep_scaler", "control_scaler"):
            if getattr(self, name) is None:
                raise ValueError(f"缺少 Scaler: {name}，请确认 training.pkl 是否正确")

    # ──────────────────── 环境质量 ────────────────────

    def env_quality(self, raw: dict) -> tuple[np.ndarray, np.ndarray]:
        """处理单条环境数据 → (numeric (6,), odor_idx (scalar)).

        raw 应包含字段:
            temp, humidity, temp_humidity_interaction,
            odor_type (str), odor_intensity, odor_duration, odor_preference
        """
        # 1. 构建数值特征向量
        numeric = np.array([raw.get(c, 0.0) for c in _ENV_NUMERIC_COLS], dtype=np.float32)
        numeric = self.env_scaler.transform(numeric.reshape(1, -1)).reshape(-1).astype(np.float32)

        # 2. 气味编码
        odor_str = raw.get("odor_type", "无")
        odor_idx = np.array(_ODOR_TYPE_MAPPING.get(odor_str, 0), dtype=np.int64)

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
            env_history: 最近 N 条环境数据（每条同 env_quality 的 raw）
            static_raw: 静态协变量字典（age, gender, bmi, ...）
            prev_sleep: 前一日睡眠指标（可选，无则填零）
            seq_len: 最大序列长度（需与训练时一致）

        Returns:
            env_seq: (seq_len, 4) 标准化环境序列（已 padding/截断）
            static: (11,) 静态特征（标准化 + 原始拼接）
            hist: (5,) 历史睡眠指标（标准化）
            seq_len_tensor: (1,) 有效时间步数
        """
        # 1. 环境序列：取最后 seq_len 条
        env_list = env_history[-seq_len:]
        raw_seq_len = len(env_list)
        env_arr = np.zeros((raw_seq_len, len(_ENV_SEQ_COLS)), dtype=np.float32)
        for t, row in enumerate(env_list):
            env_arr[t] = [row.get(c, 0.0) for c in _ENV_SEQ_COLS]

        # 标准化后用零填充到 seq_len
        # 注意：env_scaler 拟合在 6 列上（ENV_NUMERIC_COLS），而 env_seq 只用前 4 列（ENV_SEQ_COLS）。
        # 训练时 env_seq 取自已标准化的 DataFrame，故无需调用 transform()；
        # 推理时手动使用前 4 列的 mean_/scale_ 做标准化，保持与训练一致。
        if raw_seq_len > 0:
            env_arr = (
                (env_arr - self.env_scaler.mean_[:4]) / self.env_scaler.scale_[:4]
            ).astype(np.float32)

        env_seq = np.zeros((seq_len, len(_ENV_SEQ_COLS)), dtype=np.float32)
        if raw_seq_len >= seq_len:
            env_seq = env_arr[:seq_len].astype(np.float32)
        elif raw_seq_len > 0:
            env_seq[seq_len - raw_seq_len:] = env_arr.astype(np.float32)
        # raw_seq_len == 0 → env_seq 保持全零

        # 2. 静态特征
        static_vals = []
        for c in _STATIC_COLS:
            val = static_raw.get(c, 0)
            static_vals.append(float(val) if not isinstance(val, (int, float)) else val)
        static_np = np.array(static_vals, dtype=np.float32)

        # 仅对数值列做标准化，其余列保持原值
        for idx, col in enumerate(_STATIC_COLS):
            if col in _STATIC_SCALE_COLS:
                col_idx = _STATIC_SCALE_COLS.index(col)
                static_np[idx] = (
                    static_np[idx] - self.static_scaler.mean_[col_idx]
                ) / self.static_scaler.scale_[col_idx]

        # 3. 历史睡眠指标
        if prev_sleep and any(prev_sleep.get(c) is not None for c in _SLEEP_BASE_COLS):
            hist_arr = np.array([prev_sleep.get(c, 0.0) for c in _SLEEP_BASE_COLS], dtype=np.float32)
            hist_arr = self.sleep_scaler.transform(hist_arr.reshape(1, -1)).reshape(-1).astype(np.float32)
        else:
            hist_arr = np.zeros(len(_SLEEP_BASE_COLS), dtype=np.float32)

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
            state_history: 最近 N 条状态数据，每条包含:
                temp, humidity, odor_intensity, sleep_stage, time_of_day
            seq_len: 最大序列长度（需与训练时一致）

        Returns:
            state_seq: (seq_len, 5)
            seq_len_tensor: (1,)
        """
        state_list = state_history[-seq_len:]
        raw_seq_len = len(state_list)
        state_arr = np.zeros((raw_seq_len, len(_CONTROL_STATE_COLS)), dtype=np.float32)
        for t, row in enumerate(state_list):
            state_arr[t] = [row.get(c, 0.0) for c in _CONTROL_STATE_COLS]

        if raw_seq_len > 0:
            state_arr = self.control_scaler.transform(state_arr).astype(np.float32)

        state_seq = np.zeros((seq_len, len(_CONTROL_STATE_COLS)), dtype=np.float32)
        if raw_seq_len >= seq_len:
            state_seq = state_arr[:seq_len].astype(np.float32)
        elif raw_seq_len > 0:
            state_seq[seq_len - raw_seq_len:] = state_arr.astype(np.float32)

        seq_len_tensor = np.array(max(1, min(raw_seq_len, seq_len)), dtype=np.int64)
        return state_seq, seq_len_tensor
