#!/usr/bin/env python3
"""
预测脚本：使用训练好的模型进行推理
支持单样本预测、批量评估、可读结果输出

维度参数从 [`ModelConfig`](config.py) 读取，也可从 checkpoint 目录自动加载。
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from config import ModelConfig, get_default_config
from sleep_model import (
    EnvQualityClassifier,
    SleepImpactPredictor,
    ControlPolicyModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────── 可读结果格式化 ────────────────────────────

# 映射字典（与 config.py 中的语义对齐）
_CLASS_LABELS = ["优", "良", "中", "差"]
_RISK_LABELS = ["无风险", "过热风险", "过冷风险", "气味不适风险"]
_SLEEP_METRIC_LABELS = [
    "睡眠效率",
    "入睡潜伏期",
    "深睡时长",
    "觉醒次数",
    "呼吸暂停低通气指数",
    "主观睡眠质量",
]
_SLEEP_METRIC_UNITS = ["%", "分钟", "小时", "次", "", "分"]
_DISCRETE_ACTION_LABELS = ["香薰开关", "空调开关", "风扇开关"]
_CONTINUOUS_ACTION_LABELS = ["温度调节幅度", "湿度调节幅度"]


def _format_env_quality(results: Dict[str, np.ndarray]) -> List[str]:
    """格式化环境质量预测结果为可读文本。"""
    lines: List[str] = []
    for i in range(len(results["class_pred"])):
        cls_idx = int(results["class_pred"][i])
        label = _CLASS_LABELS[cls_idx] if 0 <= cls_idx < len(_CLASS_LABELS) else f"未知({cls_idx})"
        lines.append(f"环境质量分类：{label}")

    for i, c in enumerate(results["comfort"].flatten()):
        desc = "高" if c >= 0.7 else ("中" if c >= 0.4 else "低")
        lines.append(f"舒适度：{c:.2%}（{desc}）")

    risk_probs = results["risk_probs"]
    for i in range(risk_probs.shape[0]):
        risks = [f"{_RISK_LABELS[j]}{risk_probs[i, j]:.1%}" for j in range(risk_probs.shape[1])]
        lines.append(f"风险概率：{'，'.join(risks)}")

    class_probs = results["class_probs"]
    for i in range(class_probs.shape[0]):
        probs = [f"{_CLASS_LABELS[j]}{class_probs[i, j]:.1%}" for j in range(class_probs.shape[1])]
        lines.append(f"分类概率：{'，'.join(probs)}")

    return lines


def _fmt(val: float, unit: str) -> str:
    """数值转可读字符串，带单位。"""
    if unit == "%":
        return f"{val * 100:.1f}%"
    if unit == "分钟":
        return f"{val:.0f}分钟"
    if unit == "小时":
        return f"{val:.1f}小时"
    if unit == "次":
        return f"{val:.0f}次"
    if unit == "分":
        return f"{val:.2f}分"
    return f"{val:.4f}"


def _format_sleep_impact(predictions: np.ndarray) -> List[str]:
    """格式化睡眠影响预测结果为可读文本。"""
    lines: List[str] = []
    for i in range(predictions.shape[0]):
        metrics = []
        for j in range(predictions.shape[1]):
            val = predictions[i, j]
            unit = _SLEEP_METRIC_UNITS[j] if j < len(_SLEEP_METRIC_UNITS) else ""
            metrics.append(f"{_SLEEP_METRIC_LABELS[j]}：{_fmt(val, unit)}")
        lines.append("，".join(metrics))
    return lines


def _format_control_policy(results: Dict[str, np.ndarray]) -> List[str]:
    """格式化控制策略预测结果为可读文本。"""
    lines: List[str] = []
    for i in range(len(results["discrete_action"])):
        act_idx = int(results["discrete_action"][i])
        label = (
            _DISCRETE_ACTION_LABELS[act_idx]
            if 0 <= act_idx < len(_DISCRETE_ACTION_LABELS)
            else f"未知({act_idx})"
        )
        lines.append(f"离散动作：{label}")

    cont_actions = results["continuous_action"]
    for i in range(cont_actions.shape[0]):
        n_cont = min(cont_actions.shape[1], len(_CONTINUOUS_ACTION_LABELS))
        actions = [
            f"{_CONTINUOUS_ACTION_LABELS[j]}{cont_actions[i, j]:+.2f}"
            for j in range(n_cont)
        ]
        lines.append(f"连续动作（{'，'.join(actions)}）")

    for i, sv in enumerate(results["state_value"].flatten()):
        lines.append(f"状态价值：{sv:.4f}")

    if "log_prob" in results:
        for i, lp in enumerate(results["log_prob"].flatten()):
            lines.append(f"动作对数概率：{lp:.4f}")

    return lines


def format_predictions(
    model_type: str,
    results: Dict[str, np.ndarray] | np.ndarray,
) -> str:
    """将预测结果转为可读文本。"""
    if model_type == "env_quality":
        assert isinstance(results, dict)
        lines = _format_env_quality(results)
    elif model_type == "sleep_impact":
        assert isinstance(results, np.ndarray)
        lines = _format_sleep_impact(results)
    elif model_type == "control_policy":
        assert isinstance(results, dict)
        lines = _format_control_policy(results)
    else:
        lines = [f"未知模型类型：{model_type}"]
    return "\n".join(lines)


def load_model(
    model_type: str,
    checkpoint_path: str,
    device: str,
    config: Optional[ModelConfig] = None,
):
    """加载指定类型的模型及权重。

    Args:
        model_type: 模型类型
        checkpoint_path: 权重文件路径
        device: 推理设备
        config: 模型配置。若为 None，使用默认配置。
    """
    if config is None:
        config = get_default_config()

    if model_type == "env_quality":
        eq = config.env_quality
        model = EnvQualityClassifier(
            numeric_dim=eq.numeric_dim,
            odor_vocab_size=eq.odor_vocab_size,
            odor_emb_dim=eq.odor_emb_dim,
            hidden_dim=eq.hidden_dim,
            risk_dim=eq.risk_dim,
            num_classes=eq.num_classes,
        )
    elif model_type == "sleep_impact":
        si = config.sleep_impact
        model = SleepImpactPredictor(
            env_seq_dim=si.env_seq_dim,
            static_dim=si.static_dim,
            hist_dim=si.hist_dim,
            lstm_hidden_dim=si.lstm_hidden_dim,
            lstm_layers=si.lstm_layers,
            fusion_hidden_dim=si.fusion_hidden_dim,
            output_dim=si.output_dim,
            dropout=0.0,
        )
    elif model_type == "control_policy":
        cp = config.control_policy
        model = ControlPolicyModel(
            state_dim=cp.state_dim,
            discrete_action_dim=cp.discrete_action_dim,
            continuous_action_dim=cp.continuous_action_dim,
            hidden_dim=cp.hidden_dim,
            rnn_layers=cp.rnn_layers,
            rnn_type=cp.rnn_type,
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    logger.info(f"模型 {model_type} 已加载权重: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_env_quality(model, numeric: np.ndarray, odor: np.ndarray) -> Dict[str, np.ndarray]:
    """预测环境质量"""
    device = next(model.parameters()).device
    numeric = torch.FloatTensor(numeric).to(device)
    odor = torch.LongTensor(odor).to(device)
    if numeric.dim() == 1:
        numeric = numeric.unsqueeze(0)

    out = model(numeric, odor)
    comfort = out["comfort_score"].cpu().numpy()
    risk_probs = torch.softmax(out["risk_logits"], dim=-1).cpu().numpy()
    class_probs = torch.softmax(out["class_logits"], dim=-1).cpu().numpy()
    class_pred = np.argmax(class_probs, axis=-1)

    return {
        "comfort": comfort,
        "risk_probs": risk_probs,
        "class_probs": class_probs,
        "class_pred": class_pred,
    }


@torch.no_grad()
def predict_sleep_impact(
    model,
    env_seq: np.ndarray,
    static: np.ndarray,
    history: np.ndarray,
    seq_lengths: np.ndarray = None,
) -> np.ndarray:
    """预测睡眠指标"""
    device = next(model.parameters()).device
    env_seq = torch.FloatTensor(env_seq).to(device)
    static = torch.FloatTensor(static).to(device)
    history = torch.FloatTensor(history).to(device)
    if env_seq.dim() == 2:
        env_seq = env_seq.unsqueeze(0)
        static = static.unsqueeze(0)
        history = history.unsqueeze(0)
    if seq_lengths is not None:
        seq_lengths = torch.LongTensor(seq_lengths).to(device)
    else:
        seq_lengths = None

    outputs = model(env_seq, static, history, seq_lengths)
    return outputs.cpu().numpy()


@torch.no_grad()
def predict_control(
    model,
    state_seq: np.ndarray,
    seq_lengths: np.ndarray = None,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """控制策略预测（生成动作）"""
    device = next(model.parameters()).device
    state_seq = torch.FloatTensor(state_seq).to(device)
    if state_seq.dim() == 2:
        state_seq = state_seq.unsqueeze(0)
    if seq_lengths is not None:
        seq_lengths = torch.LongTensor(seq_lengths).to(device)
    else:
        seq_lengths = None

    act_out = model.act(state_seq, seq_lengths, deterministic=deterministic)
    disc_action = act_out["discrete_action"]
    cont_action = act_out["continuous_action"]
    state_value = act_out["state_value"]

    result = {
        "discrete_action": disc_action.cpu().numpy(),
        "continuous_action": cont_action.cpu().numpy(),
        "state_value": state_value.cpu().numpy(),
    }
    if "log_prob" in act_out:
        result["log_prob"] = act_out["log_prob"].cpu().numpy()
    return result


def main():
    parser = argparse.ArgumentParser(description="模型预测脚本")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["env_quality", "sleep_impact", "control_policy"],
        help="模型类型",
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="模型权重路径")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--input", type=str, required=True, help="输入数据文件（.npz 格式）")
    parser.add_argument("--output", type=str, default=None, help="结果保存路径")
    parser.add_argument(
        "--deterministic", action="store_true",
        help="控制策略是否使用确定性动作（默认随机采样）",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="model_config.json 路径（可选，默认从 checkpoint 目录自动加载）",
    )
    args = parser.parse_args()

    # 加载配置
    from pathlib import Path
    if args.config:
        config = ModelConfig.load(args.config)
    else:
        auto_path = Path(args.checkpoint).parent / "model_config.json"
        if auto_path.exists():
            config = ModelConfig.load(str(auto_path))
            logger.info(f"自动加载配置: {auto_path}")
        else:
            config = get_default_config()
            logger.info("使用默认配置")

    # 加载模型
    model = load_model(args.model, args.checkpoint, args.device, config)

    # 读取输入数据
    data = dict(np.load(args.input, allow_pickle=True))

    # 根据模型类型进行推理
    if args.model == "env_quality":
        numeric = data["numeric"]
        odor = data["odor"]
        results = predict_env_quality(model, numeric, odor)
    elif args.model == "sleep_impact":
        env_seq = data["env_seq"]
        static = data["static"]
        history = data["history"]
        seq_lengths = data.get("seq_lengths", None)
        results = predict_sleep_impact(model, env_seq, static, history, seq_lengths)
    else:  # control_policy
        state_seq = data["state_seq"]
        seq_lengths = data.get("seq_lengths", None)
        results = predict_control(model, state_seq, seq_lengths, args.deterministic)

    # 输出
    readable = format_predictions(args.model, results)

    if args.output:
        if isinstance(results, dict):
            np.savez(args.output, **results)
        else:
            np.save(args.output, results)
        logger.info(f"结果已保存至 {args.output}")
        print("\n" + readable)
    else:
        logger.info("预测结果：")
        print("─" * 40)
        print(readable)
        print("─" * 40)
        if isinstance(results, dict):
            for k, v in results.items():
                print(f"  {k}: {v}")
        else:
            print("  raw:", results)


if __name__ == "__main__":
    main()
