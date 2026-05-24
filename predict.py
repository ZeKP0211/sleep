#!/usr/bin/env python3
"""
预测脚本：使用训练好的模型进行推理
支持单样本预测、批量评估、可读结果输出
"""

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from sleep_model import (
    EnvQualityClassifier,
    SleepImpactPredictor,
    ControlPolicyModel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ──────────────────────────── 可读结果格式化 ────────────────────────────

# 映射字典与 sleep_model.py 参数注释严格对齐
# sleep_model.py L23: num_classes=4 → 优、良、中、差
_CLASS_LABELS = ["优", "良", "中", "差"]
# sleep_model.py L22: risk_dim=4 → 无风险、过热、过冷、气味不适（与标签 0-3 对齐）
_RISK_LABELS = ["无风险", "过热风险", "过冷风险", "气味不适风险"]
# sleep_model.py L309: output_dim=6
_SLEEP_METRIC_LABELS = [
    "睡眠效率",
    "入睡潜伏期",
    "深睡时长",
    "觉醒次数",
    "呼吸暂停低通气指数",
    "主观睡眠质量",
]
_SLEEP_METRIC_UNITS = ["%", "分钟", "小时", "次", "", "分"]
# sleep_model.py L314: discrete_action_dim=3 → 开/关香薰，开/关空调，开/关风扇
_DISCRETE_ACTION_LABELS = ["香薰开关", "空调开关", "风扇开关"]
# sleep_model.py L315: continuous_action_dim=2 → 温度调节幅度，湿度调节幅度
_CONTINUOUS_ACTION_LABELS = ["温度调节幅度", "湿度调节幅度"]


def _format_env_quality(results: Dict[str, np.ndarray]) -> List[str]:
    """格式化环境质量预测结果为可读文本。"""
    lines: List[str] = []
    # class_pred
    for i in range(len(results["class_pred"])):
        cls_idx = int(results["class_pred"][i])
        label = _CLASS_LABELS[cls_idx] if 0 <= cls_idx < len(_CLASS_LABELS) else f"未知({cls_idx})"
        lines.append(f"环境质量分类：{label}")

    # comfort
    for i, c in enumerate(results["comfort"].flatten()):
        desc = "高" if c >= 0.7 else ("中" if c >= 0.4 else "低")
        lines.append(f"舒适度：{c:.2%}（{desc}）")

    # risk_probs（4 类 softmax：无风险、过热、过冷、气味不适）
    risk_probs = results["risk_probs"]
    for i in range(risk_probs.shape[0]):
        risks = [f"{_RISK_LABELS[j]}{risk_probs[i, j]:.1%}" for j in range(risk_probs.shape[1])]
        lines.append(f"风险概率：{'，'.join(risks)}")

    # class_probs
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
            unit = _SLEEP_METRIC_UNITS[j]
            metrics.append(f"{_SLEEP_METRIC_LABELS[j]}：{_fmt(val, unit)}")
        lines.append("，".join(metrics))
    return lines


def _format_control_policy(results: Dict[str, np.ndarray]) -> List[str]:
    """格式化控制策略预测结果为可读文本。"""
    lines: List[str] = []
    for i in range(len(results["discrete_action"])):
        act_idx = int(results["discrete_action"][i])
        label = _DISCRETE_ACTION_LABELS[act_idx] if 0 <= act_idx < len(_DISCRETE_ACTION_LABELS) else f"未知({act_idx})"
        lines.append(f"离散动作：{label}")

    cont_actions = results["continuous_action"]
    for i in range(cont_actions.shape[0]):
        actions = [
            f"{_CONTINUOUS_ACTION_LABELS[j]}{cont_actions[i, j]:+.2f}"
            for j in range(cont_actions.shape[1])
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
    """将预测结果转为可读文本。

    Args:
        model_type: 模型类型（env_quality / sleep_impact / control_policy）
        results: 预测结果（dict 或 ndarray）

    Returns:
        可读文本字符串
    """
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


def load_model(model_type: str, checkpoint_path: str, device: str):
    """加载指定类型的模型及权重"""
    if model_type == "env_quality":
        model = EnvQualityClassifier(
            numeric_dim=6,
            odor_vocab_size=5,
            odor_emb_dim=4,
            hidden_dim=64,
            risk_dim=4,
            num_classes=4,
        )
    elif model_type == "sleep_impact":
        model = SleepImpactPredictor(
            env_seq_dim=4,
            static_dim=11,
            hist_dim=5,
            lstm_hidden_dim=64,
            lstm_layers=2,
            fusion_hidden_dim=128,
            output_dim=6,
            dropout=0.0,
        )
    elif model_type == "control_policy":
        model = ControlPolicyModel(
            state_dim=5,
            discrete_action_dim=3,
            continuous_action_dim=2,
            hidden_dim=128,
            rnn_layers=2,
            rnn_type="GRU",
        )
    else:
        raise ValueError(f"未知模型类型: {model_type}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)  # 直接是state_dict
    model.to(device)
    model.eval()
    logger.info(f"模型 {model_type} 已加载权重: {checkpoint_path}")
    return model


@torch.no_grad()
def predict_env_quality(model, numeric: np.ndarray, odor: np.ndarray) -> Dict[str, np.ndarray]:
    """
    预测环境质量
    Args:
        numeric: shape (batch, 6) 或 (6,)
        odor: shape (batch,) 或标量
    Returns:
        comfort (0~1),
        risk_probs (softmax over 4 classes: 无风险/过热/过冷/气味不适),
        class_probs (softmax over 4 classes: 优/良/中/差)
    """
    device = next(model.parameters()).device
    numeric = torch.FloatTensor(numeric).to(device)
    odor = torch.LongTensor(odor).to(device)
    if numeric.dim() == 1:
        numeric = numeric.unsqueeze(0)
    # Embedding 接受 (batch,) 或标量，不需要 unsqueeze

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
    """
    预测睡眠指标
    Args:
        env_seq: (batch, T, 4)
        static: (batch, 11)
        history: (batch, 5)
        seq_lengths: (batch,) 或 None
    Returns:
        预测值数组 (batch, 6)，顺序:
        [睡眠效率, 入睡潜伏期, 深睡时长, 觉醒次数, 呼吸暂停指数, 主观睡眠质量]
    """
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

    outputs = model(env_seq, static, history, seq_lengths)  # (batch, 6)
    return outputs.cpu().numpy()


@torch.no_grad()
def predict_control(
    model,
    state_seq: np.ndarray,
    seq_lengths: np.ndarray = None,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    """
    控制策略预测（生成动作）
    Args:
        state_seq: (batch, T, 5) 或 (T, 5)
        seq_lengths: (batch,) 或 None
        deterministic: True 时使用均值（连续动作），离散动作取 argmax
    Returns:
        discrete_action, continuous_action, state_value
    """
    device = next(model.parameters()).device
    state_seq = torch.FloatTensor(state_seq).to(device)
    if state_seq.dim() == 2:
        state_seq = state_seq.unsqueeze(0)
    if seq_lengths is not None:
        seq_lengths = torch.LongTensor(seq_lengths).to(device)
    else:
        seq_lengths = None

    if deterministic:
        out = model.forward(state_seq, seq_lengths)
        disc_action = torch.argmax(out["discrete_logits"], dim=-1)
        cont_action = out["continuous_mean"]
        state_value = out["state_value"]
        log_prob = None
    else:
        act_out = model.act(state_seq, seq_lengths)
        disc_action = act_out["discrete_action"]
        cont_action = act_out["continuous_action"]
        state_value = act_out["state_value"]
        log_prob = act_out["log_prob"].cpu().numpy()

    result = {
        "discrete_action": disc_action.cpu().numpy(),
        "continuous_action": cont_action.cpu().numpy(),
        "state_value": state_value.cpu().numpy(),
    }
    if log_prob is not None:
        result["log_prob"] = log_prob
    return result


def main():
    parser = argparse.ArgumentParser(description="模型预测脚本")
    parser.add_argument("--model", type=str, required=True,
                        choices=["env_quality", "sleep_impact", "control_policy"],
                        help="模型类型")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--input", type=str, required=True,
                        help="输入数据文件（.npz 格式）")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径（.npy 或 .npz），若不指定则打印")
    parser.add_argument("--deterministic", action="store_true",
                        help="控制策略是否使用确定性动作（默认随机采样）")
    args = parser.parse_args()

    # 加载模型
    model = load_model(args.model, args.checkpoint, args.device)

    # 读取输入数据
    data = dict(np.load(args.input, allow_pickle=True))

    # 根据模型类型进行推理
    if args.model == "env_quality":
        numeric = data["numeric"]  # (N, 6) 或 (6,)
        odor = data["odor"]        # (N,) 或标量
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
        # 同时打印可读结果
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