# 气味温湿度睡眠模型

## 项目结构

| 文件 | 说明 |
|------|------|
| `sleep_model.py` | 三个模型定义：环境质量分类器、睡眠影响预测器、控制策略模型 |
| `train.py` | 训练脚本 |
| `predict.py` | 预测脚本（基于 `.npz` 输入文件） |
| `demo_predict.py` | 预测演示：自动从 CSV 生成 `.npz` 并提示预测命令 |
| `losses.py` | 损失函数 |
| `generate_data.py` | 生成模拟测试数据 |

## 使用流程

### 1. 生成数据
```bash
python train.py --generate-data --data-dir data --sample-users 20 --sample-days 3
```

### 2. 训练模型
```bash
python train.py --data-dir data --epochs 5 --batch-size 16 --seq-len 6
```
训练完成后，模型权重保存在 `data/checkpoints/` 目录。

### 3. 运行预测

使用 predict.py

**环境质量预测**（舒适度/风险/分类）：
```bash
python predict.py --model env_quality --checkpoint data/checkpoints/env_quality_best.pth --input input_env.npz
```

**睡眠影响预测**（次日睡眠指标）：
```bash
python predict.py --model sleep_impact --checkpoint data/checkpoints/sleep_prediction_best.pth --input input_sleep.npz
```

**控制策略预测**（设备动作建议）：
```bash
python predict.py --model control_policy --checkpoint data/checkpoints/control_policy_best.pth --input input_ctrl.npz
```

添加 `--output result.npz` 保存结果，添加 `--cpu` 强制使用 CPU。

## 预测输出说明

| 模型 | 输出字段 | 说明 |
|------|----------|------|
| `env_quality` | `comfort` | 舒适度分数 (0~1) |
| | `risk_probs` | 过热/过冷/气味不适风险概率 |
| | `class_probs` | 环境质量分类概率 [舒适, 过热, 过冷, 气味差] |
| | `class_pred` | 预测类别索引 |
| `sleep_impact` | 6维数组 | [睡眠效率, 入睡潜伏期, 深睡时长, 觉醒次数, 呼吸暂停指数, 主观睡眠质量] |
| `control_policy` | `discrete_action` | 离散动作（开关类设备） |
| | `continuous_action` | 连续动作（调节幅度） |
| | `state_value` | 状态价值估计 |


# 训练（生成 preprocessing.pkl + checkpoints）
python train.py --generate-data --data-dir data --sample-users 100 --sample-days 30 --epochs 20

# 导出 ONNX
python export_onnx.py --checkpoint-dir data/checkpoints

# 启动服务
python -m inference.api --checkpoint-dir data/checkpoints --port 8000
