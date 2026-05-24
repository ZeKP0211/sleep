#!/usr/bin/env python3
"""ONNX 模型导出脚本。

将训练好的 PyTorch 模型导出为 ONNX 格式，供 ONNX Runtime 推理使用。

用法::

    # 导出所有模型
    python export_onnx.py --checkpoint-dir data/checkpoints --output-dir data/checkpoints

    # 仅导出环境质量模型
    python export_onnx.py --checkpoint-dir data/checkpoints --output-dir data/checkpoints --model env_quality
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from sleep_model import EnvQualityClassifier, SleepImpactPredictor, ControlPolicyModel


# ── 模型创建（参数与训练/build_models 一致） ──

def _create_env_quality() -> EnvQualityClassifier:
    return EnvQualityClassifier(
        numeric_dim=6, odor_vocab_size=5, odor_emb_dim=4, hidden_dim=64,
        risk_dim=4, num_classes=4,
    )


def _create_sleep_impact() -> SleepImpactPredictor:
    return SleepImpactPredictor(
        env_seq_dim=4, static_dim=11, hist_dim=5,
        lstm_hidden_dim=64, lstm_layers=2, fusion_hidden_dim=128,
        output_dim=6, dropout=0.0,
    )


def _create_control_policy() -> ControlPolicyModel:
    return ControlPolicyModel(
        state_dim=5, discrete_action_dim=3, continuous_action_dim=2,
        hidden_dim=128, rnn_layers=2, rnn_type="GRU",
    )


_MODEL_REGISTRY = {
    "env_quality": (_create_env_quality, "env_quality_best.pth"),
    "sleep_impact": (_create_sleep_impact, "sleep_prediction_best.pth"),
    "control_policy": (_create_control_policy, "control_policy_best.pth"),
}


# ── 导出函数 ──

def export_env_quality(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    """导出环境质量模型为 ONNX。

    输入: numeric_feat (batch, 6), odor_idx (batch,)
    输出: class_logits, risk_logits, comfort_score
    """
    model = _create_env_quality()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_numeric = torch.randn(1, 6, device=device)
    dummy_odor = torch.zeros(1, dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        (dummy_numeric, dummy_odor),
        output_path,
        input_names=["numeric_feat", "odor_idx"],
        output_names=["comfort_score", "risk_logits", "class_logits"],
        dynamic_axes={
            "numeric_feat": {0: "batch"},
            "odor_idx": {0: "batch"},
            "class_logits": {0: "batch"},
            "risk_logits": {0: "batch"},
            "comfort_score": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"[env_quality] 已导出: {output_path}")


def export_sleep_impact(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    seq_len: int = 24,
    opset_version: int = 17,
) -> None:
    """导出睡眠影响预测模型为 ONNX。

    输入: env_seq (batch, seq_len, 4), static_features (batch, 11),
          history_features (batch, 5), seq_lengths (batch,) 可选
    输出: sleep_metrics (batch, 6)
    """
    model = _create_sleep_impact()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # 导出两个版本：带 seq_lengths 和不带
    dummy_env = torch.randn(1, seq_len, 4, device=device)
    dummy_static = torch.randn(1, 11, device=device)
    dummy_hist = torch.randn(1, 5, device=device)
    dummy_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)

    # 使用 tuple 形式 (env_seq, static_features, history_features, seq_lengths)
    torch.onnx.export(
        model,
        (dummy_env, dummy_static, dummy_hist, dummy_seq_len),
        output_path,
        input_names=["env_seq", "static_features", "history_features", "seq_lengths"],
        output_names=["sleep_metrics"],
        dynamic_axes={
            "env_seq": {0: "batch", 1: "seq_len"},
            "static_features": {0: "batch"},
            "history_features": {0: "batch"},
            "seq_lengths": {0: "batch"},
            "sleep_metrics": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"[sleep_impact] 已导出: {output_path}")


def export_control_policy(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu",
    seq_len: int = 24,
    opset_version: int = 17,
) -> None:
    """导出控制策略模型为 ONNX（仅 forward，不含采样）。

    输入: state_seq (batch, seq_len, 5), seq_lengths (batch,) 可选
    输出: discrete_logits, continuous_mean, continuous_log_std, state_value
    """
    model = _create_control_policy()
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_state = torch.randn(1, seq_len, 5, device=device)
    dummy_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)

    torch.onnx.export(
        model,
        (dummy_state, dummy_seq_len),
        output_path,
        input_names=["state_seq", "seq_lengths"],
        output_names=["discrete_logits", "continuous_mean", "continuous_log_std", "state_value"],
        dynamic_axes={
            "state_seq": {0: "batch", 1: "seq_len"},
            "seq_lengths": {0: "batch"},
            "discrete_logits": {0: "batch"},
            "continuous_mean": {0: "batch"},
            "continuous_log_std": {0: "batch"},
            "state_value": {0: "batch"},
        },
        opset_version=opset_version,
        dynamo=False,
    )
    print(f"[control_policy] 已导出: {output_path}")


# ── CLI ──

def main():
    parser = argparse.ArgumentParser(description="ONNX 模型导出")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="模型权重所在目录")
    parser.add_argument("--output-dir", type=str, default=None, help="ONNX 输出目录（默认同 checkpoint-dir）")
    parser.add_argument("--model", type=str, default=None,
                        choices=list(_MODEL_REGISTRY.keys()),
                        help="仅导出指定模型（默认导出全部）")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=24, help="序列长度（需与训练时一致）")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    args = parser.parse_args()

    out_dir = Path(args.output_dir or args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    export_fns = {
        "env_quality": export_env_quality,
        "sleep_impact": export_sleep_impact,
        "control_policy": export_control_policy,
    }

    models_to_export = [args.model] if args.model else list(_MODEL_REGISTRY.keys())

    for name in models_to_export:
        create_fn, ckpt_name = _MODEL_REGISTRY[name]
        ckpt_path = Path(args.checkpoint_dir) / ckpt_name
        if not ckpt_path.exists():
            print(f"[{name}] 跳过: 权重文件不存在 ({ckpt_path})")
            continue

        if name == "sleep_impact":
            export_sleep_impact(str(ckpt_path), str(out_dir / "sleep_impact.onnx"),
                               args.device, args.seq_len, args.opset)
        elif name == "control_policy":
            export_control_policy(str(ckpt_path), str(out_dir / "control_policy.onnx"),
                                 args.device, args.seq_len, args.opset)
        else:
            export_env_quality(str(ckpt_path), str(out_dir / "env_quality.onnx"),
                               args.device, args.opset)

    print("导出完成。")


if __name__ == "__main__":
    main()
