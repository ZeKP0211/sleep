# 睡眠环境智能调控系统

基于深度强化学习的智能睡眠环境调控系统，集成了**环境质量评估**、**睡眠影响预测**和**设备控制策略**三大模型。

## 架构概览

```
┌─────────────────────────────────────────────────────────┐
│              ControlPolicyModel (PPO Actor-Critic)      │
│  输入: 环境状态序列 (温湿度/气味/睡眠阶段/时间)           │
│  输出: 离散动作(香薰/空调/风扇) + 连续动作(温湿度调节)   │
│         + 状态价值 V(s)                                 │
├─────────────────────────────────────────────────────────┤
│              EnvQualityClassifier                       │
│  输入: 环境数值特征 + 气味类型                            │
│  输出: 舒适度分数 / 风险概率 / 环境质量分类               │
├─────────────────────────────────────────────────────────┤
│              SleepImpactPredictor                       │
│  输入: 环境序列 + 静态特征 + 历史睡眠                     │
│  输出: 睡眠效率/入睡潜伏期/深睡时长/觉醒次数/...          │
└─────────────────────────────────────────────────────────┘
```

**关键特性：**
- Actor-Critic 架构 + PPO Clipped 目标 + GAE 优势估计
- 状态依赖的策略方差（而非全局固定 log_std）
- 支持确定性推理(部署) / 随机采样(训练/探索)
- ONNX 导出 + ONNX Runtime 高性能推理
- FastAPI 微服务 + 完整推理管线

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成样本数据 + 训练

```bash
# 完整训练 (环境质量 + 睡眠预测 + PPO 控制策略)
python train.py --generate-data --sample-users 20 --sample-days 3 --epochs 5 --ppo-epochs 10 --seq-len 6

# 仅使用 CPU
python train.py --generate-data --cpu

# 自定义 PPO 超参数
python train.py --generate-data --ppo-clip-ratio 0.15 --ppo-gamma 0.99 --ppo-lam 0.95 --ppo-rollout-steps 64
```

训练后产出物在 `data/checkpoints/`：
| 文件 | 说明 |
|------|------|
| `env_quality_best.pth` | 环境质量分类器权重 |
| `sleep_prediction_best.pth` | 睡眠影响预测器权重 |
| `actor_critic_best.pth` | PPO Actor-Critic 控制策略权重 |
| `preprocessing.pkl` | 标准化器 (Scaler) |
| `model_config.json` | 模型配置 |

### 3. 导出 ONNX

```bash
python export_onnx.py --checkpoint-dir data/checkpoints --seq-len 6
```

### 4. 启动推理服务

```bash
# 方式 1: 独立启动
cd inference && python api.py --checkpoint-dir ../data/checkpoints --port 8000

# 方式 2: uvicorn
uvicorn inference.api:create_app --factory --host 0.0.0.0 --port 8000
```

### 5. 测试 API

```bash
python test_api.py                          # 测试 localhost:8000
python test_api.py --base-url http://1.2.3.4:8000
```

### 6. 命令行推理 (不需要启动服务)

```bash
# 环境质量预测
python predict.py --model env_quality --checkpoint data/checkpoints/env_quality_best.pth --input sample.npz

# 控制策略 (确定性)
python predict.py --model control_policy --checkpoint data/checkpoints/actor_critic_best.pth --input sample.npz --deterministic

# 控制策略 (随机采样)
python predict.py --model control_policy --checkpoint data/checkpoints/actor_critic_best.pth --input sample.npz
```

## API 接口

### GET `/v1/health` — 健康检查

```json
{"status": "ok", "models": ["env_quality", "sleep_impact", "control_policy"]}
```

### POST `/v1/predict/env-quality` — 环境质量

```json
// 请求
{"temp": 26.5, "humidity": 55.0, "odor_type": "薰衣草", "odor_intensity": 0.3, ...}
// 响应
{"comfort": 0.6543, "risk_probs": [0.7, 0.2, 0.05, 0.05], "class_label": "良", ...}
```

### POST `/v1/predict/sleep-impact` — 睡眠影响

```json
// 请求
{"env_history": [...], "static_features": {...}, "prev_sleep": {...}}
// 响应
{"sleep_efficiency": 0.85, "sleep_latency": 12.3, "deep_sleep_duration": 2.1, ...}
```

### POST `/v1/predict/control-policy` — 设备控制策略

```json
// 请求
{
  "state_history": [
    {"temp": 26.5, "humidity": 55.0, "odor_intensity": 0.3, "sleep_stage": 2, "time_of_day": 22}
  ],
  "deterministic": true   // true=确定性, false=随机采样
}
// 响应
{
  "discrete_action": 1,
  "discrete_action_label": "空调开关",
  "continuous_action": [0.5, 1.2],
  "state_value": 3.456
}
```

## PPO 训练超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--ppo-clip-ratio` | 0.2 | PPO clipping ε |
| `--ppo-value-coef` | 0.5 | 价值损失权重 |
| `--ppo-entropy-coef` | 0.01 | 熵奖励权重 |
| `--ppo-gamma` | 0.99 | GAE 折扣因子 |
| `--ppo-lam` | 0.95 | GAE λ |
| `--ppo-rollout-steps` | 32 | 每次更新的 rollout 步数 |
| `--ppo-epochs` | 10 | PPO 训练轮数 |

## 项目结构

```
sleep/
├── sleep_model.py          # 三个模型定义 (EnvQuality / SleepImpact / ControlPolicy AC)
├── losses.py               # 损失函数 (交叉熵 / SmoothL1 / GAE / PPO clipped)
├── config.py               # 集中配置 (维度 / 超参数 / 数据列名 / PPO 参数)
├── train.py                # 训练主脚本 (监督训练 + PPO rollout+GAE)
├── generate_data.py        # 样本数据生成 (含 reward 列)
├── predict.py              # PyTorch 推理脚本
├── export_onnx.py          # ONNX 导出
├── test_api.py             # API 全端点测试
├── inference/
│   ├── api.py              # FastAPI 推理服务
│   ├── model_loader.py     # ONNX Runtime 模型加载 + 采样逻辑
│   └── preprocessing.py    # 推理预处理管线
└── requirements.txt
```

## 技术栈

- **RL**: PPO (Proximal Policy Optimization) + GAE (Generalized Advantage Estimation)
- **模型**: PyTorch + RNN (GRU/LSTM) + Actor-Critic
- **部署**: ONNX Runtime + FastAPI
- **动作空间**: 混合 (3 个离散动作 + 2 个连续动作)