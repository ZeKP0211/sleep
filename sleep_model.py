import torch
import torch.nn as nn
from typing import Optional
from torch.distributions import Categorical, Normal


class EnvQualityClassifier(nn.Module):
    """环境质量评估模型（多任务环境质量评估）

    该模型同时输出：
      - 舒适度分数（0-1）
      - 风险类别概率（无风险、过热、过冷、气味不适）
      - 环境质量分类（优、良、中、差）
    """

    def __init__(
        self,
        numeric_dim: int = 6,  # 温度、湿度、交互项、强度、时长、喜好度
        odor_vocab_size: int = 5,  # 气味类别数（薰衣草、沉香、川芎、无等）
        odor_emb_dim: int = 4,  #将每种气味映射为一个长度为 4 的向量
        hidden_dim: int = 64,   #共享层中的隐藏单元数
        risk_dim: int = 4,  #4 类风险（无风险、过热、过冷、气味不适），与标签 0-3 对齐
        num_classes: int = 4,   #传统单分类任务的类别数量（将环境质量分为：优、良、中、差
    ):
        super().__init__()
        self.odor_embedding = nn.Embedding(odor_vocab_size, odor_emb_dim)
        input_dim = numeric_dim + odor_emb_dim

        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hidden_dim),
        )

        self.comfort_head = nn.Linear(hidden_dim, 1)
        self.risk_head = nn.Linear(hidden_dim, risk_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes)

    def forward(self, numeric_feat: torch.Tensor, odor_idx: torch.Tensor):
        odor_emb = self.odor_embedding(odor_idx)  
        x = torch.cat([numeric_feat, odor_emb], dim=-1)
        shared = self.shared_net(x)

        comfort_score = torch.sigmoid(self.comfort_head(shared)).squeeze(-1)
        risk_logits = self.risk_head(shared)
        class_logits = self.class_head(shared)

        return {
            "comfort_score": comfort_score,
            "risk_logits": risk_logits,
            "class_logits": class_logits,
        }  


class SleepImpactPredictor(nn.Module):
    """睡眠影响预测模型

    组合多模态输入：
      - 环境时间序列特征（温度、湿度、气味强度）
      - 静态协变量（年龄、性别、BMI、季节、健康、生活习惯）
      - 历史睡眠指标（睡眠效率、入睡潜伏期、深睡时长、觉醒次数等）

    输出：睡眠质量预测，例如睡眠效率、入睡潜伏期、深睡时长、觉醒次数、主观睡眠质量。
    """

    def __init__(
        self,
        env_seq_dim: int,
        static_dim: int,
        hist_dim: int,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        fusion_hidden_dim: int = 128,
        output_dim: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.env_lstm = nn.LSTM(
            input_size=env_seq_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=False,
        )

        self.static_encoder = nn.Sequential(
            nn.Linear(static_dim, static_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(static_dim * 2, static_dim),
            nn.ReLU(inplace=True),
        )

        self.history_encoder = nn.Sequential(
            nn.Linear(hist_dim, hist_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hist_dim * 2, hist_dim),
            nn.ReLU(inplace=True),
        )

        self.fusion = nn.Sequential(
            nn.Linear(lstm_hidden_dim + static_dim + hist_dim, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, output_dim),
        )

        # 添加注意力池化
        self.attn_query = nn.Linear(lstm_hidden_dim, 1)
        self.output_dim = output_dim

    def forward(
        self,
        env_seq: torch.Tensor,
        static_features: torch.Tensor,
        history_features: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # LSTM前向
        if seq_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                env_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out_packed, _ = self.env_lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out_packed, batch_first=True)
        else:
            lstm_out, _ = self.env_lstm(env_seq) 

        # 注意力池化
        attn_scores = self.attn_query(lstm_out).squeeze(-1) 
        if seq_lengths is not None:
            # 对padding位置做mask，避免无效时间步参与注意力分配
            max_len = lstm_out.size(1)
            mask = (
                torch.arange(max_len, device=lstm_out.device)
                .unsqueeze(0)
                .expand(lstm_out.size(0), -1)
            ) >= seq_lengths.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)  
        env_vec = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)  

        static_out = self.static_encoder(static_features)
        hist_out = self.history_encoder(history_features)
        fused = torch.cat([env_vec, static_out, hist_out], dim=-1)
        out = self.fusion(fused)

        # 输出顺序：
        # [睡眠效率(0-1), 入睡潜伏期(>0), 深睡时长(>0), 觉醒次数(>0),
        #  呼吸暂停低通气指数(>0), 主观睡眠质量(0-1)]
        constrained = torch.stack(
            [
                torch.sigmoid(out[:, 0]),  # 睡眠效率 0~1
                torch.relu(out[:, 1]),  # 入睡潜伏期 >0
                torch.relu(out[:, 2]),  # 深睡时长 >0
                torch.relu(out[:, 3]),  # 觉醒次数 >0
                torch.relu(out[:, 4]),  # 呼吸暂停低通气指数 >0
                torch.sigmoid(out[:, 5]),  # 主观睡眠质量 0~1
            ],
            dim=-1,
        )
        return constrained


class ControlPolicyModel(nn.Module):
    """控制策略模型（Actor-Critic）

    输入状态：当前时间、温湿度、气味、睡眠阶段、历史行为/设备状态等。
    输出：
      Actor（策略）：离散动作 logits + 连续动作分布参数
      Critic（价值）：状态价值 V(s)
    """

    _RNN_MAP = {"GRU": nn.GRU, "LSTM": nn.LSTM}

    def __init__(
        self,
        state_dim: int,
        discrete_action_dim: int,
        continuous_action_dim: int,
        hidden_dim: int = 128,
        rnn_layers: int = 2,
        rnn_type: str = "GRU",
        action_log_std_init: float = -0.5,
    ):
        super().__init__()
        rnn_cls = self._RNN_MAP.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type='{rnn_type}', expected 'GRU' or 'LSTM'")
        self.rnn = rnn_cls(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=0.2 if rnn_layers > 1 else 0.0,
        )

        # Actor heads
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)
        self.continuous_mean_head = nn.Linear(hidden_dim, continuous_action_dim)
        self.continuous_log_std = nn.Parameter(
            torch.full((continuous_action_dim,), action_log_std_init)
        )

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

    def _encode_state(
        self, state_seq: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # state_seq: (batch, seq_len, state_dim)
        if seq_lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                state_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.rnn(packed)
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
            last_idx = (seq_lengths - 1).clamp(min=0).to(rnn_out.device)
            final_hidden = rnn_out[torch.arange(rnn_out.size(0), device=rnn_out.device), last_idx]
        else:
            rnn_out, _ = self.rnn(state_seq)
            final_hidden = rnn_out[:, -1, :]
        return final_hidden

    def forward(
        self, state_seq: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        final_hidden = self._encode_state(state_seq, seq_lengths)
        discrete_logits = self.discrete_head(final_hidden)
        continuous_mean = self.continuous_mean_head(final_hidden)
        continuous_log_std = self.continuous_log_std.unsqueeze(0).expand_as(continuous_mean)
        state_value = self.value_head(final_hidden).squeeze(-1)

        return {
            "discrete_logits": discrete_logits,
            "continuous_mean": continuous_mean,
            "continuous_log_std": continuous_log_std,
            "state_value": state_value,
        }

    def _make_dist(self, out: dict[str, torch.Tensor]):
        """从 forward 输出创建离散 + 连续分布。"""
        discrete_dist = Categorical(logits=out["discrete_logits"])
        continuous_std = torch.exp(out["continuous_log_std"])
        continuous_dist = Normal(out["continuous_mean"], continuous_std)
        return discrete_dist, continuous_dist

    def act(
        self, state_seq: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        out = self.forward(state_seq, seq_lengths)
        discrete_dist, continuous_dist = self._make_dist(out)

        discrete_action = discrete_dist.sample()
        continuous_action = continuous_dist.sample()
        log_prob = discrete_dist.log_prob(discrete_action) + continuous_dist.log_prob(
            continuous_action
        ).sum(dim=-1)
        entropy = discrete_dist.entropy() + continuous_dist.entropy().sum(dim=-1)

        return {**out, "discrete_action": discrete_action, "continuous_action": continuous_action, "log_prob": log_prob, "entropy": entropy}

    def evaluate_actions(
        self,
        state_seq: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """评估给定动作的 log_prob / entropy / value，用于 PPO 损失。"""
        out = self.forward(state_seq, seq_lengths)
        discrete_dist, continuous_dist = self._make_dist(out)

        log_prob = discrete_dist.log_prob(discrete_actions) + continuous_dist.log_prob(
            continuous_actions
        ).sum(dim=-1)
        entropy = discrete_dist.entropy() + continuous_dist.entropy().sum(dim=-1)

        return {**out, "log_prob": log_prob, "entropy": entropy}


def build_models() -> dict:
    """构建默认模型"""
    models = {
        "env_quality_classifier": EnvQualityClassifier(
            numeric_dim=6,
            odor_vocab_size=5,
            odor_emb_dim=4,
            hidden_dim=64,
            risk_dim=4,
            num_classes=4,
        ),
        "sleep_impact_predictor": SleepImpactPredictor(
            env_seq_dim=4,
            static_dim=11,  # age, gender, bmi, season, health_nose, health_asthma, health_depression, habit_alcohol, habit_caffeine, habit_exercise, habit_screen_time
            hist_dim=5,
            lstm_hidden_dim=64,
            lstm_layers=2,
            fusion_hidden_dim=128,
            output_dim=6,  # sleep_efficiency, sleep_latency, deep_sleep_duration, awakenings, apnea_index, subjective_sleep_quality
            dropout=0.2,
        ),
        "control_policy_model": ControlPolicyModel(
            state_dim=5,  # temp, humidity, odor_intensity, sleep_stage, time_of_day
            discrete_action_dim=3,  # 例如：开/关香薰，开/关空调，开/关风扇
            continuous_action_dim=2,  # 例如：温度调节幅度，湿度调节幅度
            hidden_dim=128,
            rnn_layers=2,
            rnn_type="GRU",
        ),
    }
    return models


