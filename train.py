"""训练主脚本。

用法::

    # 默认配置训练
    python train.py --generate-data

    # 自定义模型维度（无需修改代码）
    python train.py --generate-data --env-hidden-dim 128 --sleep-hidden-dim 256

    # 从 JSON 配置文件加载
    python train.py --generate-data --config my_config.json
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

from config import (
    DataConfig,
    ModelConfig,
    get_default_config,
    get_default_data_config,
)
from losses import control_policy_losses, env_quality_loss, sleep_impact_loss
from sleep_model import build_models


# ── 数据路径配置 ──

@dataclass
class DatasetPaths:
    env_data_path: str
    static_data_path: str
    sleep_history_path: str
    env_labels_path: Optional[str] = None
    control_data_path: Optional[str] = None


# ── 工具函数 ──

def _ensure_file(path: str, name: str) -> None:
    if not Path(path).exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _safe_numeric_fill(df: pd.DataFrame, cols: tuple[str, ...]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)


def _to_device(batch, device):
    """将数据 batch 中的 Tensor 移至目标设备."""
    return tuple(x.to(device) if isinstance(x, torch.Tensor) else x for x in batch)


# ── 数据集 ──

class SleepDataset(Dataset):
    """睡眠数据集。

    所有列名与维度从 [`DataConfig`](config.py) 读取，避免硬编码。
    """

    def __init__(
        self,
        paths: DatasetPaths,
        seq_len: int = 24,
        task: str = "env_quality",
        data_config: Optional[DataConfig] = None,
    ):
        if task not in {"env_quality", "sleep_prediction", "control_policy"}:
            raise ValueError("task must be one of: env_quality, sleep_prediction, control_policy")

        _ensure_file(paths.env_data_path, "env_data")
        _ensure_file(paths.static_data_path, "static_data")
        _ensure_file(paths.sleep_history_path, "sleep_history")

        self.task = task
        self.seq_len = seq_len
        self.cfg = data_config or get_default_data_config()

        self.env_data = pd.read_csv(paths.env_data_path).sort_values(["user_id", "timestamp"])
        self.static_data = pd.read_csv(paths.static_data_path).set_index("user_id")
        self.sleep_history = pd.read_csv(paths.sleep_history_path).sort_values(["user_id", "date"])
        self.env_labels = None
        self.control_data = None

        if task == "env_quality":
            if not paths.env_labels_path:
                raise ValueError("env_labels_path is required for env_quality task")
            _ensure_file(paths.env_labels_path, "env_labels")
            self.env_labels = pd.read_csv(paths.env_labels_path).set_index(["user_id", "timestamp"])

        if task == "control_policy":
            if not paths.control_data_path:
                raise ValueError("control_data_path is required for control_policy task")
            _ensure_file(paths.control_data_path, "control_data")
            self.control_data = pd.read_csv(paths.control_data_path).sort_values(["user_id", "timestamp"])

        self._preprocess()

    def _preprocess(self) -> None:
        """数据预处理：编码 + 标准化。

        从 [`DataConfig`](config.py) 获取所有列名与映射。
        """
        # 气味编码
        self.env_data["odor_type_encoded"] = (
            self.env_data.get("odor_type", pd.Series(["无"] * len(self.env_data)))
            .map(self.cfg.odor_type_mapping)
            .fillna(0)
            .astype(int)
        )

        _safe_numeric_fill(self.env_data, self.cfg.env_numeric_cols)
        _safe_numeric_fill(self.static_data, self.cfg.static_cols)
        _safe_numeric_fill(self.sleep_history, self.cfg.sleep_base_cols)

        if "subjective_sleep_quality" not in self.sleep_history.columns:
            self.sleep_history["subjective_sleep_quality"] = np.float32(0.5)
        _safe_numeric_fill(self.sleep_history, ("subjective_sleep_quality",))

        # 环境标准化器
        self.env_scaler = StandardScaler()
        env_num_cols = list(self.cfg.env_numeric_cols)
        self.env_data[env_num_cols] = self.env_scaler.fit_transform(
            self.env_data[env_num_cols]
        ).astype(np.float32)

        # 静态协变量标准化器（仅对数值列）
        static_scale_cols = [
            c for c in self.cfg.static_scale_cols if c in self.static_data.columns
        ]
        self.static_scaler = StandardScaler()
        if static_scale_cols:
            self.static_data[static_scale_cols] = self.static_scaler.fit_transform(
                self.static_data[static_scale_cols]
            ).astype(np.float32)

        # 睡眠指标标准化器
        self.sleep_scaler = StandardScaler()
        sleep_base_cols = list(self.cfg.sleep_base_cols)
        self.sleep_history[sleep_base_cols] = self.sleep_scaler.fit_transform(
            self.sleep_history[sleep_base_cols]
        ).astype(np.float32)

        # 控制策略标准化器
        if self.control_data is not None:
            ctrl_state_cols = list(self.cfg.control_state_cols)
            ctrl_all_cols = ctrl_state_cols + list(self.cfg.control_cont_action_cols) + ["action_discrete"]
            _safe_numeric_fill(self.control_data, tuple(ctrl_all_cols))
            self.control_scaler = StandardScaler()
            self.control_data[ctrl_state_cols] = self.control_scaler.fit_transform(
                self.control_data[ctrl_state_cols]
            ).astype(np.float32)
        else:
            self.control_scaler = None

    @property
    def scalers(self) -> dict:
        """返回所有已拟合的 Scaler 字典，键名与 inference 模块一致."""
        result = {}
        if hasattr(self, "env_scaler") and self.env_scaler is not None:
            result["env_scaler"] = self.env_scaler
        if hasattr(self, "static_scaler") and self.static_scaler is not None:
            result["static_scaler"] = self.static_scaler
        if hasattr(self, "sleep_scaler") and self.sleep_scaler is not None:
            result["sleep_scaler"] = self.sleep_scaler
        if hasattr(self, "control_scaler") and self.control_scaler is not None:
            result["control_scaler"] = self.control_scaler
        return result

    def __len__(self) -> int:
        if self.task == "env_quality":
            return len(self.env_data)
        if self.task == "sleep_prediction":
            return len(self.sleep_history)
        assert self.control_data is not None
        return max(len(self.control_data) - self.seq_len + 1, 0)

    def __getitem__(self, idx: int):
        if self.task == "env_quality":
            row = self.env_data.iloc[idx]
            numeric_feat = torch.tensor(
                row[list(self.cfg.env_numeric_cols)].values.astype(np.float32), dtype=torch.float32
            )
            odor_idx = torch.tensor(int(row["odor_type_encoded"]), dtype=torch.long)
            assert self.env_labels is not None
            label = torch.tensor(
                int(self.env_labels.loc[(row["user_id"], row["timestamp"]), "env_quality_label"]),
                dtype=torch.long,
            )
            return numeric_feat, odor_idx, label

        if self.task == "sleep_prediction":
            row = self.sleep_history.iloc[idx]
            user_id = row["user_id"]
            env_seq_cols = list(self.cfg.env_seq_cols)
            user_env = self.env_data[self.env_data["user_id"] == user_id].tail(self.seq_len)
            env_values = user_env[env_seq_cols].values.astype(np.float32)
            seq_len_raw = len(env_values)
            if seq_len_raw < self.seq_len:
                pad = np.zeros((self.seq_len - seq_len_raw, len(env_seq_cols)), dtype=np.float32)
                env_values = np.vstack([pad, env_values])

            env_seq = torch.tensor(env_values, dtype=torch.float32)
            static_feat = torch.tensor(
                self.static_data.loc[user_id, list(self.cfg.static_cols)].values.astype(np.float32),
                dtype=torch.float32,
            )
            sleep_base_cols = list(self.cfg.sleep_base_cols)
            prev_sleep = self.sleep_history[
                (self.sleep_history["user_id"] == user_id)
                & (self.sleep_history["date"] < row["date"])
            ].tail(1)
            if len(prev_sleep) > 0:
                hist_feat = torch.tensor(
                    prev_sleep[sleep_base_cols].values[0].astype(np.float32), dtype=torch.float32
                )
            else:
                hist_feat = torch.zeros(len(sleep_base_cols), dtype=torch.float32)

            target = torch.tensor(
                row[list(self.cfg.sleep_target_cols)].values.astype(np.float32), dtype=torch.float32
            )
            seq_len_tensor = torch.tensor(max(1, min(seq_len_raw, self.seq_len)), dtype=torch.long)
            return env_seq, static_feat, hist_feat, seq_len_tensor, target

        assert self.control_data is not None
        ctrl_state_cols = list(self.cfg.control_state_cols)
        ctrl_cont_cols = list(self.cfg.control_cont_action_cols)
        seq_data = self.control_data.iloc[idx: idx + self.seq_len]
        state_seq = torch.tensor(
            seq_data[ctrl_state_cols].values.astype(np.float32), dtype=torch.float32
        )
        action_discrete = torch.tensor(int(seq_data["action_discrete"].values[-1]), dtype=torch.long)
        action_cont = torch.tensor(
            seq_data[ctrl_cont_cols].values[-1].astype(np.float32), dtype=torch.float32
        )
        return state_seq, torch.tensor(self.seq_len, dtype=torch.long), action_discrete, action_cont


# ── DataLoader 创建 ──

def create_data_loaders(
    paths: DatasetPaths,
    batch_size: int = 32,
    seq_len: int = 24,
    train_split: float = 0.8,
    data_config: Optional[DataConfig] = None,
) -> tuple[dict[str, DataLoader], dict[str, SleepDataset]]:
    """为所有可用任务创建 DataLoader，同时返回 dataset 引用以获取 scaler."""
    loaders = {}
    datasets = {}
    for task, required_attr in [
        ("env_quality", "env_labels_path"),
        ("sleep_prediction", None),
        ("control_policy", "control_data_path"),
    ]:
        if required_attr and getattr(paths, required_attr) is None:
            continue
        dataset = SleepDataset(paths=paths, seq_len=seq_len, task=task, data_config=data_config)
        if len(dataset) < 2:
            continue
        train_size = int(len(dataset) * train_split)
        if train_size == 0:
            continue
        train_ds, val_ds = random_split(dataset, [train_size, len(dataset) - train_size])
        loaders[f"{task}_train"] = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        loaders[f"{task}_val"] = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        datasets[task] = dataset
    return loaders, datasets


# ── 通用训练循环 ──

def _train_epoch(model, loader, optimizer, device, forward_fn):
    """运行一个训练 epoch，返回平均 loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = _to_device(batch, device)
        optimizer.zero_grad()
        loss = forward_fn(model, batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _eval_epoch(model, loader, device, forward_fn):
    """运行一个验证 epoch，返回平均 loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            total_loss += forward_fn(model, batch).item()
    return total_loss / len(loader)


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    forward_fn,
    num_epochs: int = 10,
    device: str = "cpu",
    lr: float = 1e-3,
    checkpoint_path: Optional[Path] = None,
    tag: str = "model",
    extra_metrics_fn=None,
) -> None:
    """通用训练循环."""
    model.to(device)
    if checkpoint_path and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded existing best checkpoint from {checkpoint_path}")

    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        train_loss = _train_epoch(model, train_loader, optimizer, device, forward_fn)
        val_loss = _eval_epoch(model, val_loader, device, forward_fn)

        if checkpoint_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        log = f"[{tag}][{epoch+1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
        if extra_metrics_fn:
            log += extra_metrics_fn(model, val_loader, device)
        print(log)


# ── 各任务前向传播函数 ──

def _env_forward(model, batch):
    numeric_feat, odor_idx, labels = batch
    return env_quality_loss(model(numeric_feat, odor_idx), labels)


def _env_metrics(model, val_loader, device):
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            numeric_feat, odor_idx, labels = _to_device(batch, device)
            pred = model(numeric_feat, odor_idx)["class_logits"].argmax(dim=1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    return f" val_acc={100 * correct / max(total, 1):.2f}%"


def _sleep_forward(model, batch):
    env_seq, static_feat, hist_feat, seq_lengths, targets = batch
    return sleep_impact_loss(
        model(env_seq, static_feat, hist_feat, seq_lengths=seq_lengths), targets
    )


def _control_forward(entropy_coef):
    def fn(model, batch):
        state_seq, seq_lengths, action_discrete, action_cont = batch
        eval_out = model.evaluate_actions(
            state_seq, action_discrete, action_cont, seq_lengths=seq_lengths
        )
        return control_policy_losses(eval_out, action_discrete, action_cont, entropy_coef=entropy_coef)[
            "loss"
        ]

    return fn


# ── 参数解析 ──

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sleep training pipeline")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--generate-data", action="store_true")
    parser.add_argument("--sample-users", type=int, default=20)
    parser.add_argument("--sample-days", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="JSON 配置文件路径（可覆盖默认模型维度）",
    )

    # ── 模型维度覆盖参数（不加前缀则全部覆盖） ──
    dim_group = parser.add_argument_group("模型维度覆盖 (ModelConfig)")
    dim_group.add_argument("--env-hidden-dim", type=int, default=None, help="环境质量模型 hidden_dim")
    dim_group.add_argument("--env-risk-dim", type=int, default=None, help="环境质量模型 risk_dim")
    dim_group.add_argument("--env-num-classes", type=int, default=None, help="环境质量模型 num_classes")
    dim_group.add_argument("--env-odor-vocab-size", type=int, default=None)
    dim_group.add_argument("--env-odor-emb-dim", type=int, default=None)

    dim_group.add_argument("--sleep-lstm-hidden-dim", type=int, default=None)
    dim_group.add_argument("--sleep-lstm-layers", type=int, default=None)
    dim_group.add_argument("--sleep-fusion-hidden-dim", type=int, default=None)
    dim_group.add_argument("--sleep-dropout", type=float, default=None)

    dim_group.add_argument("--ctrl-hidden-dim", type=int, default=None)
    dim_group.add_argument("--ctrl-rnn-layers", type=int, default=None)
    dim_group.add_argument("--ctrl-rnn-type", type=str, default=None, choices=["GRU", "LSTM"])

    return parser


def _apply_cli_overrides(config: ModelConfig, args: argparse.Namespace) -> ModelConfig:
    """将命令行维度参数覆盖到 ModelConfig."""
    eq = config.env_quality
    si = config.sleep_impact
    cp = config.control_policy

    if args.env_hidden_dim is not None:
        eq.hidden_dim = args.env_hidden_dim
    if args.env_risk_dim is not None:
        eq.risk_dim = args.env_risk_dim
    if args.env_num_classes is not None:
        eq.num_classes = args.env_num_classes
    if args.env_odor_vocab_size is not None:
        eq.odor_vocab_size = args.env_odor_vocab_size
    if args.env_odor_emb_dim is not None:
        eq.odor_emb_dim = args.env_odor_emb_dim

    if args.sleep_lstm_hidden_dim is not None:
        si.lstm_hidden_dim = args.sleep_lstm_hidden_dim
    if args.sleep_lstm_layers is not None:
        si.lstm_layers = args.sleep_lstm_layers
    if args.sleep_fusion_hidden_dim is not None:
        si.fusion_hidden_dim = args.sleep_fusion_hidden_dim
    if args.sleep_dropout is not None:
        si.dropout = args.sleep_dropout

    if args.ctrl_hidden_dim is not None:
        cp.hidden_dim = args.ctrl_hidden_dim
    if args.ctrl_rnn_layers is not None:
        cp.rnn_layers = args.ctrl_rnn_layers
    if args.ctrl_rnn_type is not None:
        cp.rnn_type = args.ctrl_rnn_type

    return config


# ── 训练入口 ──

def run_training(args: argparse.Namespace) -> None:
    # 构建配置
    if args.config:
        model_config = ModelConfig.load(args.config)
        print(f"已加载配置文件: {args.config}")
    else:
        model_config = get_default_config()
    model_config = _apply_cli_overrides(model_config, args)
    data_config = get_default_data_config()

    # 同步 DataConfig 推导维度到 ModelConfig（确保一致性）
    model_config.env_quality.numeric_dim = data_config.env_numeric_dim
    model_config.sleep_impact.env_seq_dim = data_config.env_seq_dim
    model_config.sleep_impact.static_dim = data_config.static_dim
    model_config.sleep_impact.hist_dim = data_config.hist_dim
    model_config.sleep_impact.output_dim = data_config.output_dim
    model_config.control_policy.state_dim = data_config.state_dim
    model_config.control_policy.continuous_action_dim = data_config.continuous_action_dim

    # 生成样本数据
    if args.generate_data:
        from generate_data import generate_sample_data

        generate_sample_data(
            num_users=args.sample_users,
            days_per_user=args.sample_days,
            output_dir=args.data_dir,
        )

    paths = DatasetPaths(
        env_data_path=str(Path(args.data_dir) / "env_features.csv"),
        static_data_path=str(Path(args.data_dir) / "static_covariates.csv"),
        sleep_history_path=str(Path(args.data_dir) / "sleep_history.csv"),
        env_labels_path=str(Path(args.data_dir) / "env_labels.csv"),
        control_data_path=str(Path(args.data_dir) / "control_data.csv"),
    )

    checkpoint_dir = Path(args.data_dir) / "checkpoints"
    _ensure_dir(checkpoint_dir)

    loaders, datasets = create_data_loaders(
        paths=paths,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        train_split=args.train_split,
        data_config=data_config,
    )
    models = build_models(model_config)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    print(f"使用设备: {device}")

    # 任务配置列表
    task_configs = [
        (
            "env_quality",
            models["env_quality_classifier"],
            _env_forward,
            "env_quality",
            "env",
            _env_metrics,
        ),
        (
            "sleep_prediction",
            models["sleep_impact_predictor"],
            _sleep_forward,
            "sleep_prediction",
            "sleep",
            None,
        ),
        (
            "control_policy",
            models["control_policy_model"],
            _control_forward(args.entropy_coef),
            "control_policy",
            "actor_critic",
            None,
        ),
    ]

    for loader_key, model, forward_fn, ckpt_name, tag, metrics_fn in task_configs:
        train_key = f"{loader_key}_train"
        val_key = f"{loader_key}_val"
        if train_key in loaders:
            _train_model(
                model=model,
                train_loader=loaders[train_key],
                val_loader=loaders[val_key],
                forward_fn=forward_fn,
                num_epochs=args.epochs,
                device=device,
                lr=args.lr,
                checkpoint_path=checkpoint_dir / f"{ckpt_name}_best.pth",
                tag=tag,
                extra_metrics_fn=metrics_fn,
            )

    # 持久化 Scaler 工件
    import joblib
    from collections import OrderedDict

    all_scalers = OrderedDict()
    for task, ds in datasets.items():
        for name, scaler in ds.scalers.items():
            all_scalers[name] = scaler
    if all_scalers:
        scaler_path = checkpoint_dir / "preprocessing.pkl"
        joblib.dump(all_scalers, scaler_path)
        print(f"Scaler 工件已保存至 {scaler_path}（共 {len(all_scalers)} 个）")

    # 持久化模型配置（供后续导出 / 推理使用）
    config_path = checkpoint_dir / "model_config.json"
    model_config.save(str(config_path))
    print(f"模型配置已保存至 {config_path}")

    print("训练完成。")


if __name__ == "__main__":
    run_training(build_arg_parser().parse_args())
