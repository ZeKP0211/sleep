"""预测演示：从 CSV 数据到模型推理"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def prepare_env_quality_input(csv_dir: str, output: str, sample_idx: int = 0):
    """从 CSV 提取环境质量预测输入并保存为 .npz"""
    env = pd.read_csv(Path(csv_dir) / "env_features.csv")
    labels = pd.read_csv(Path(csv_dir) / "env_labels.csv").set_index(["user_id", "timestamp"])

    row = env.iloc[sample_idx]
    numeric = np.array(row[["temp", "humidity", "temp_humidity_interaction",
                             "body_temp", "body_humidity",
                             "odor_intensity", "odor_duration", "odor_preference"]],
                       dtype=np.float32)

    odor_type_mapping = {"无": 0, "薰衣草": 1, "沉香": 2, "川芎": 3, "其他": 4}
    odor = np.array([odor_type_mapping.get(row.get("odor_type", "无"), 0)], dtype=np.int64)

    label = int(labels.loc[(row["user_id"], row["timestamp"]), "env_quality_label"])

    np.savez(output, numeric=numeric, odor=odor, label=label)
    print(f"→ 已保存环境质量输入: {output}")
    print(f"  用户={row['user_id']}, 标签={label}")


def prepare_sleep_impact_input(csv_dir: str, output: str, seq_len: int = 24):
    """从 CSV 提取睡眠影响预测输入并保存为 .npz"""
    env = pd.read_csv(Path(csv_dir) / "env_features.csv").sort_values(["user_id", "timestamp"])
    static = pd.read_csv(Path(csv_dir) / "static_covariates.csv").set_index("user_id")
    sleep = pd.read_csv(Path(csv_dir) / "sleep_history.csv").sort_values(["user_id", "date"])

    # 取第一个用户的最后一条睡眠记录
    user_id = sleep["user_id"].iloc[-1]
    row = sleep[sleep["user_id"] == user_id].iloc[-1]

    user_env = env[env["user_id"] == user_id].tail(seq_len)
    env_values = user_env[["temp", "humidity", "temp_humidity_interaction", "odor_intensity"]].values.astype(np.float32)
    seq_len_raw = len(env_values)
    if seq_len_raw < seq_len:
        pad = np.zeros((seq_len - seq_len_raw, 4), dtype=np.float32)
        env_values = np.vstack([pad, env_values])

    static_feat = static.loc[user_id, ["age", "gender", "bmi", "season",
                                       "health_nose", "health_asthma", "health_depression",
                                       "habit_alcohol", "habit_caffeine", "habit_exercise",
                                       "habit_screen_time"]].values.astype(np.float32)

    prev = sleep[(sleep["user_id"] == user_id) & (sleep["date"] < row["date"])]
    if len(prev) > 0:
        hist_feat = prev[["sleep_efficiency", "sleep_latency", "deep_sleep_duration",
                          "awakenings", "apnea_index"]].values[-1].astype(np.float32)
    else:
        hist_feat = np.zeros(5, dtype=np.float32)

    target = row[["sleep_efficiency", "sleep_latency", "deep_sleep_duration",
                  "awakenings", "apnea_index", "subjective_sleep_quality"]].values.astype(np.float32)

    np.savez(output,
             env_seq=env_values[np.newaxis, :, :],       # (1, T, 4)
             static=static_feat[np.newaxis, :],           # (1, 11)
             history=hist_feat[np.newaxis, :],            # (1, 5)
             seq_lengths=np.array([min(seq_len_raw, seq_len)], dtype=np.int64),
             target=target)
    print(f"→ 已保存睡眠影响输入: {output}")
    print(f"  用户={user_id}, 日期={row['date']}")


def prepare_control_input(csv_dir: str, output: str, seq_len: int = 6):
    """从 CSV 提取控制策略预测输入并保存为 .npz"""
    ctrl = pd.read_csv(Path(csv_dir) / "control_data.csv").sort_values(["user_id", "timestamp"])

    # 取第一个用户的最后一段序列
    user_id = ctrl["user_id"].iloc[-1]
    user_ctrl = ctrl[ctrl["user_id"] == user_id].tail(seq_len)

    state_seq = user_ctrl[["temp", "humidity", "odor_intensity", "sleep_stage", "time_of_day"]].values.astype(np.float32)
    if len(state_seq) < seq_len:
        pad = np.zeros((seq_len - len(state_seq), 5), dtype=np.float32)
        state_seq = np.vstack([pad, state_seq])

    np.savez(output,
             state_seq=state_seq[np.newaxis, :, :],      # (1, T, 5)
             seq_lengths=np.array([min(len(user_ctrl), seq_len)], dtype=np.int64))
    print(f"→ 已保存控制策略输入: {output}")
    print(f"  用户={user_id}")


def main():
    parser = argparse.ArgumentParser(description="预测演示：从 CSV 生成 .npz 输入文件")
    parser.add_argument("--data-dir", default="data", help="CSV 数据目录")
    parser.add_argument("--model", choices=["env_quality", "sleep_impact", "control_policy"], default="env_quality")
    parser.add_argument("--seq-len", type=int, default=24)
    parser.add_argument("--output", default=None, help="输出 .npz 路径（默认自动生成）")
    args = parser.parse_args()

    if args.output is None:
        args.output = f"input_{args.model}.npz"

    if args.model == "env_quality":
        prepare_env_quality_input(args.data_dir, args.output)
    elif args.model == "sleep_impact":
        prepare_sleep_impact_input(args.data_dir, args.output, seq_len=args.seq_len)
    elif args.model == "control_policy":
        prepare_control_input(args.data_dir, args.output, seq_len=min(args.seq_len, 6))

    print(f"\n运行预测: python predict.py --model {args.model} --checkpoint data/checkpoints/{args.model.replace('_impact', '_prediction')}_best.pth --input {args.output}")
    print(f"示例:")
    print(f"  python predict.py --model {args.model} --checkpoint data/checkpoints/{args.model.replace('_impact', '_prediction')}_best.pth --input {args.output}")


if __name__ == "__main__":
    main()
