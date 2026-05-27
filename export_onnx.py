#!/usr/bin/env python3
"""ONNX 模型导出脚本。

将训练好的 PyTorch 模型导出为 ONNX 格式，供 ONNX Runtime 推理使用。
模型维度从 [`ModelConfig`](config.py) 读取，与训练时保持一致。

用法::

    # 导出所有模型（自动从 checkpoint 目录读取 model_config.json）
    python export_onnx.py --checkpoint-dir data/checkpoints

    # 仅导出环境质量模型
    python export_onnx.py --checkpoint-dir data/checkpoints --model env_quality
"""

import argparse
from pathlib import Path
from typing import Optional

import torch

from config import ModelConfig
from sleep_model import EnvQualityClassifier, SleepImpactPredictor, ControlPolicyModel


# ── 模型创建（从配置读取所有维度） ──

def _create_env_quality(config: ModelConfig) -> EnvQualityClassifier:
    eq = config.env_quality
    return EnvQualityClassifier(
        numeric_dim=eq.numeric_dim,
        odor_vocab_size=eq.odor_vocab_size,
        odor_emb_dim=eq.odor_emb_dim,
        hidden_dim=eq.hidden_dim,
        risk_dim=eq.risk_dim,
        num_classes=eq.num_classes,
    )


def _create_sleep_impact(config: ModelConfig) -> SleepImpactPredictor:
    si = config.sleep_impact
    return SleepImpactPredictor(
        env_seq_dim=si.env_seq_dim,
        static_dim=si.static_dim,
        hist_dim=si.hist_dim,
        lstm_hidden_dim=si.lstm_hidden_dim,
        lstm_layers=si.lstm_layers,
        fusion_hidden_dim=si.fusion_hidden_dim,
        output_dim=si.output_dim,
        dropout=0.0,  # 推理时关闭 dropout
    )


def _create_control_policy(config: ModelConfig) -> ControlPolicyModel:
    cp = config.control_policy
    return ControlPolicyModel(
        state_dim=cp.state_dim,
        discrete_action_dim=cp.discrete_action_dim,
        continuous_action_dim=cp.continuous_action_dim,
        hidden_dim=cp.hidden_dim,
        rnn_layers=cp.rnn_layers,
        rnn_type=cp.rnn_type,
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
    config: ModelConfig,
    device: str = "cpu",
    opset_version: int = 17,
) -> None:
    """导出环境质量模型为 ONNX.

    输入: numeric_feat (batch, 6), odor_idx (batch,)
    输出: comfort_score, risk_logits, class_logits
    """
    model = _create_env_quality(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_numeric = torch.randn(1, config.env_quality.numeric_dim, device=device)
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
    config: ModelConfig,
    device: str = "cpu",
    seq_len: int = 24,
    opset_version: int = 17,
) -> None:
    """导出睡眠影响预测模型为 ONNX.

    输入: env_seq (batch, seq_len, env_seq_dim), static_features (batch, static_dim),
          history_features (batch, hist_dim), seq_lengths (batch,) 可选
    输出: sleep_metrics (batch, output_dim)
    """
    si = config.sleep_impact
    model = _create_sleep_impact(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_env = torch.randn(1, seq_len, si.env_seq_dim, device=device)
    dummy_static = torch.randn(1, si.static_dim, device=device)
    dummy_hist = torch.randn(1, si.hist_dim, device=device)
    dummy_seq_len = torch.tensor([seq_len], dtype=torch.long, device=device)

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
    config: ModelConfig,
    device: str = "cpu",
    seq_len: int = 24,
    opset_version: int = 17,
) -> None:
    """导出控制策略模型为 ONNX（仅 forward，不含采样）.

    输入: state_seq (batch, seq_len, state_dim), seq_lengths (batch,) 可选
    输出: discrete_logits, continuous_mean, continuous_log_std, state_value
    """
    cp = config.control_policy
    model = _create_control_policy(config)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dummy_state = torch.randn(1, seq_len, cp.state_dim, device=device)
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

def _load_config(checkpoint_dir: str, config_path: Optional[str] = None) -> ModelConfig:
    """加载模型配置，优先级：--config > checkpoint_dir/model_config.json > 默认."""
    if config_path:
        return ModelConfig.load(config_path)

    auto_path = Path(checkpoint_dir) / "model_config.json"
    if auto_path.exists():
        print(f"自动加载配置: {auto_path}")
        return ModelConfig.load(str(auto_path))

    from config import get_default_config
    print("未找到 model_config.json，使用默认配置")
    return get_default_config()


def main():
    parser = argparse.ArgumentParser(description="ONNX 模型导出")
    parser.add_argument("--checkpoint-dir", type=str, required=True, help="模型权重所在目录")
    parser.add_argument("--output-dir", type=str, default=None, help="ONNX 输出目录（默认同 checkpoint-dir）")
    parser.add_argument(
        "--model", type=str, default=None,
        choices=list(_MODEL_REGISTRY.keys()),
        help="仅导出指定模型（默认导出全部）",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seq-len", type=int, default=24, help="序列长度（需与训练时一致）")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset 版本")
    parser.add_argument("--config", type=str, default=None, help="model_config.json 路径（可选）")
    args = parser.parse_args()

    config = _load_config(args.checkpoint_dir, args.config)

    out_dir = Path(args.output_dir or args.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_export = [args.model] if args.model else list(_MODEL_REGISTRY.keys())

    for name in models_to_export:
        _create_fn, ckpt_name = _MODEL_REGISTRY[name]
        ckpt_path = Path(args.checkpoint_dir) / ckpt_name
        if not ckpt_path.exists():
            print(f"[{name}] 跳过: 权重文件不存在 ({ckpt_path})")
            continue

        if name == "sleep_impact":
            export_sleep_impact(
                str(ckpt_path), str(out_dir / "sleep_impact.onnx"),
                config, args.device, args.seq_len, args.opset,
            )
        elif name == "control_policy":
            export_control_policy(
                str(ckpt_path), str(out_dir / "control_policy.onnx"),
                config, args.device, args.seq_len, args.opset,
            )
        else:
            export_env_quality(
                str(ckpt_path), str(out_dir / "env_quality.onnx"),
                config, args.device, args.opset,
            )

    print("导出完成。")


if __name__ == "__main__":
    main()
