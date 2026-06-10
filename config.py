"""集中配置模块。

所有模型维度、数据列名、气味映射等配置均在此定义，
其他模块从此处导入，避免硬编码散落。

用法::

    from config import ModelConfig, DataConfig, get_default_config

    # 使用默认配置
    cfg = get_default_config()
    model = EnvQualityClassifier(**cfg.env_quality.to_dict())

    # 命令行覆盖
    cfg.env_quality.hidden_dim = 128

    # 保存/加载配置
    cfg.save("data/checkpoints/config.json")
    cfg2 = ModelConfig.load("data/checkpoints/config.json")
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════
# 子模型配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class EnvQualityConfig:
    """环境质量评估模型配置。

    对应 [`sleep_model.EnvQualityClassifier`](sleep_model.py:7)。
    """

    numeric_dim: int = 8
    """数值特征维度: 温度、湿度、交互项、人体表面温度、人体表面湿度、气味强度、持续时长、喜好度"""

    odor_vocab_size: int = 5
    """气味类别数: 薰衣草、沉香、川芎、无、其他"""

    odor_emb_dim: int = 4
    """气味 Embedding 维度"""

    hidden_dim: int = 64
    """共享隐藏层维度"""

    risk_dim: int = 4
    """风险类别数: 无风险、过热、过冷、气味不适"""

    num_classes: int = 4
    """环境质量分类数: 优、良、中、差"""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SleepImpactConfig:
    """睡眠影响预测模型配置。

    对应 [`sleep_model.SleepImpactPredictor`](sleep_model.py:58)。
    """

    env_seq_dim: int = 4
    """环境序列特征维度: temp, humidity, temp_humidity_interaction, odor_intensity"""

    static_dim: int = 11
    """静态协变量维度: age, gender, bmi, season, health_nose, health_asthma,
       health_depression, habit_alcohol, habit_caffeine, habit_exercise, habit_screen_time"""

    hist_dim: int = 5
    """历史睡眠指标维度: sleep_efficiency, sleep_latency, deep_sleep_duration,
       awakenings, apnea_index"""

    lstm_hidden_dim: int = 64
    """LSTM 隐藏层维度"""

    lstm_layers: int = 2
    """LSTM 层数"""

    fusion_hidden_dim: int = 128
    """融合层隐藏维度"""

    output_dim: int = 6
    """输出维度: 睡眠效率、入睡潜伏期、深睡时长、觉醒次数、呼吸暂停指数、主观睡眠质量"""

    dropout: float = 0.2
    """Dropout 比率"""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ControlPolicyConfig:
    """控制策略模型配置。

    对应 [`sleep_model.ControlPolicyModel`](sleep_model.py:174)。
    """

    state_dim: int = 5
    """状态维度: temp, humidity, odor_intensity, sleep_stage, time_of_day"""

    discrete_action_dim: int = 3
    """离散动作维度: 香薰开关、空调开关、风扇开关"""

    continuous_action_dim: int = 2
    """连续动作维度: 温度调节幅度、湿度调节幅度"""

    hidden_dim: int = 128
    """RNN 隐藏层维度"""

    rnn_layers: int = 2
    """RNN 层数"""

    rnn_type: str = "GRU"
    """RNN 类型: 'GRU' 或 'LSTM'"""

    action_log_std_init: float = -0.5
    """连续动作对数标准差初始值"""

    # ── PPO 训练参数 ──
    ppo_clip_ratio: float = 0.2
    """PPO clipping ε"""

    ppo_value_coef: float = 0.5
    """价值损失系数"""

    ppo_entropy_coef: float = 0.01
    """熵奖励系数"""

    ppo_gamma: float = 0.99
    """GAE 折扣因子"""

    ppo_lam: float = 0.95
    """GAE λ"""

    ppo_rollout_steps: int = 32
    """每次 PPO 更新前收集的 rollout 步数"""

    ppo_epochs: int = 4
    """每轮数据上进行 PPO 更新的次数"""

    ppo_max_grad_norm: float = 0.5
    """梯度裁剪阈值"""

    def to_dict(self) -> dict:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# 顶层配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """所有模型配置的聚合容器。"""

    env_quality: EnvQualityConfig = field(default_factory=EnvQualityConfig)
    sleep_impact: SleepImpactConfig = field(default_factory=SleepImpactConfig)
    control_policy: ControlPolicyConfig = field(default_factory=ControlPolicyConfig)

    def to_dict(self) -> dict:
        return {
            "env_quality": self.env_quality.to_dict(),
            "sleep_impact": self.sleep_impact.to_dict(),
            "control_policy": self.control_policy.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(
            env_quality=EnvQualityConfig(**d.get("env_quality", {})),
            sleep_impact=SleepImpactConfig(**d.get("sleep_impact", {})),
            control_policy=ControlPolicyConfig(**d.get("control_policy", {})),
        )

    def save(self, path: str) -> None:
        """保存配置为 JSON 文件。"""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """从 JSON 文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    def clone(self) -> "ModelConfig":
        """深拷贝配置。"""
        return copy.deepcopy(self)


# ═══════════════════════════════════════════════════════════════════════════
# 数据列名配置
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DataConfig:
    """数据列名与映射的集中定义。

    所有模块从单一来源获取列名，避免 train.py 与 inference/preprocessing.py
    中的重复定义。
    """

    # ── 环境数值列 (6 列，用于 EnvQualityClassifier) ──
    env_numeric_cols: tuple[str, ...] = (
        "temp",
        "humidity",
        "temp_humidity_interaction",
        "body_temp",
        "body_humidity",
        "odor_intensity",
        "odor_duration",
        "odor_preference",
    )

    # ── 环境序列列 (4 列，用于 SleepImpactPredictor LSTM 输入) ──
    env_seq_cols: tuple[str, ...] = (
        "temp",
        "humidity",
        "temp_humidity_interaction",
        "odor_intensity",
    )

    # ── 静态协变量列 (11 列) ──
    static_cols: tuple[str, ...] = (
        "age",
        "gender",
        "bmi",
        "season",
        "health_nose",
        "health_asthma",
        "health_depression",
        "habit_alcohol",
        "habit_caffeine",
        "habit_exercise",
        "habit_screen_time",
    )

    # ── 静态协变量中需要 StandardScaler 的数值列 ──
    static_scale_cols: tuple[str, ...] = (
        "age",
        "bmi",
        "season",
        "habit_alcohol",
        "habit_caffeine",
        "habit_exercise",
        "habit_screen_time",
    )

    # ── 睡眠基础指标列 (5 列，不含主观评分) ──
    sleep_base_cols: tuple[str, ...] = (
        "sleep_efficiency",
        "sleep_latency",
        "deep_sleep_duration",
        "awakenings",
        "apnea_index",
    )

    # ── 睡眠目标列 (基础 5 列 + 主观评分) ──
    sleep_target_cols: tuple[str, ...] = (
        "sleep_efficiency",
        "sleep_latency",
        "deep_sleep_duration",
        "awakenings",
        "apnea_index",
        "subjective_sleep_quality",
    )

    # ── 控制策略状态列 (5 列) ──
    control_state_cols: tuple[str, ...] = (
        "temp",
        "humidity",
        "odor_intensity",
        "sleep_stage",
        "time_of_day",
    )

    # ── 控制策略连续动作列 (2 列) ──
    control_cont_action_cols: tuple[str, ...] = (
        "action_temp_adjust",
        "action_humidity_adjust",
    )

    # ── 气味类型映射 ──
    odor_type_mapping: dict[str, int] = field(default_factory=lambda: {
        "无": 0,
        "薰衣草": 1,
        "沉香": 2,
        "川芎": 3,
        "其他": 4,
    })

    @property
    def env_numeric_dim(self) -> int:
        return len(self.env_numeric_cols)

    @property
    def env_seq_dim(self) -> int:
        return len(self.env_seq_cols)

    @property
    def static_dim(self) -> int:
        return len(self.static_cols)

    @property
    def hist_dim(self) -> int:
        return len(self.sleep_base_cols)

    @property
    def output_dim(self) -> int:
        return len(self.sleep_target_cols)

    @property
    def state_dim(self) -> int:
        return len(self.control_state_cols)

    @property
    def continuous_action_dim(self) -> int:
        return len(self.control_cont_action_cols)

def get_default_config() -> ModelConfig:
    """返回默认 ModelConfig（所有参数使用预设值）。"""
    return ModelConfig()


def get_default_data_config() -> DataConfig:
    """返回默认 DataConfig（所有列名使用预设值）。"""
    return DataConfig()
