import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, TYPE_CHECKING
from torch.distributions import Categorical, Normal

if TYPE_CHECKING:
    from config import ModelConfig


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
    """控制策略模型（PPO Actor-Critic）

    输入状态：当前时间、温湿度、气味、睡眠阶段、历史行为/设备状态等。
    输出：
      Actor（策略）：离散动作 logits + 连续动作分布参数（状态依赖）
      Critic（价值）：状态价值 V(s)

    关键改进（相对原始版本）：
      - continuous_log_std 由状态依赖 head 输出，不再使用全局参数
      - 添加 Dropout 正则化
      - 添加 std 下界保护（softplus）
      - act() 支持 deterministic 模式
      - evaluate_actions() 返回完整的 PPO 所需字段
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
        dropout: float = 0.2,
        log_std_min: float = -5.0,
        log_std_max: float = 1.0,
    ):
        super().__init__()
        self.continuous_action_dim = continuous_action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        rnn_cls = self._RNN_MAP.get(rnn_type)
        if rnn_cls is None:
            raise ValueError(f"Unsupported rnn_type='{rnn_type}', expected 'GRU' or 'LSTM'")
        self.rnn = rnn_cls(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        # 正则化：RNN 输出后的 Dropout
        self.output_dropout = nn.Dropout(dropout)

        # Actor heads
        self.discrete_head = nn.Linear(hidden_dim, discrete_action_dim)
        self.continuous_mean_head = nn.Linear(hidden_dim, continuous_action_dim)
        # 状态依赖的 log_std（替代全局参数）
        self.continuous_log_std_head = nn.Linear(hidden_dim, continuous_action_dim)

        # Critic head
        self.value_head = nn.Linear(hidden_dim, 1)

        # 初始化 log_std head 的偏置，使其初始输出接近 action_log_std_init
        nn.init.constant_(self.continuous_log_std_head.bias, action_log_std_init)
        nn.init.zeros_(self.continuous_log_std_head.weight)

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

    def _get_log_std(self, hidden: torch.Tensor) -> torch.Tensor:
        """从隐状态计算状态依赖的连续动作 log_std，带 clamp 保护。"""
        raw = self.continuous_log_std_head(hidden)
        return torch.clamp(raw, self.log_std_min, self.log_std_max)

    def forward(
        self, state_seq: torch.Tensor, seq_lengths: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        final_hidden = self._encode_state(state_seq, seq_lengths)
        # Dropout 正则化
        final_hidden = self.output_dropout(final_hidden)

        discrete_logits = self.discrete_head(final_hidden)
        continuous_mean = self.continuous_mean_head(final_hidden)
        continuous_log_std = self._get_log_std(final_hidden)
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
        # 用 softplus 确保标准差有下界（exp(clamp 后的 log_std) 已受 clamp 保护，双保险）
        continuous_std = F.softplus(out["continuous_log_std"])
        continuous_dist = Normal(out["continuous_mean"], continuous_std)
        return discrete_dist, continuous_dist

    def act(
        self,
        state_seq: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """采样或确定性选择动作。

        Args:
            state_seq: 状态序列 (batch, seq_len, state_dim)
            seq_lengths: 各样本的有效序列长度
            deterministic: 若为 True，离散取 argmax，连续取 mean；否则采样。

        Returns:
            包含动作、分布、log_prob、entropy、value 的字典。
        """
        out = self.forward(state_seq, seq_lengths)
        discrete_dist, continuous_dist = self._make_dist(out)

        if deterministic:
            discrete_action = discrete_dist.probs.argmax(dim=-1)
            continuous_action = continuous_dist.mean
        else:
            discrete_action = discrete_dist.sample()
            continuous_action = continuous_dist.sample()

        log_prob = discrete_dist.log_prob(discrete_action) + continuous_dist.log_prob(
            continuous_action
        ).sum(dim=-1)
        entropy = discrete_dist.entropy() + continuous_dist.entropy().sum(dim=-1)

        return {
            **out,
            "discrete_action": discrete_action,
            "continuous_action": continuous_action,
            "log_prob": log_prob,
            "entropy": entropy,
        }

    def evaluate_actions(
        self,
        state_seq: torch.Tensor,
        discrete_actions: torch.Tensor,
        continuous_actions: torch.Tensor,
        seq_lengths: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """评估给定动作的 log_prob / entropy / value，用于 PPO 损失。

        Returns:
            包含 state_value, discrete_logits, continuous_mean, continuous_log_std,
            log_prob, entropy 的字典。
        """
        out = self.forward(state_seq, seq_lengths)
        discrete_dist, continuous_dist = self._make_dist(out)

        log_prob = discrete_dist.log_prob(discrete_actions) + continuous_dist.log_prob(
            continuous_actions
        ).sum(dim=-1)
        entropy = discrete_dist.entropy() + continuous_dist.entropy().sum(dim=-1)

        return {
            **out,
            "log_prob": log_prob,
            "entropy": entropy,
        }


def build_models(config: "ModelConfig | None" = None) -> dict:
    """从配置构建模型。

    Args:
        config: 模型配置。若为 None，使用默认配置。

    Returns:
        dict: 包含三个模型的字典，键名与 train.py 保持一致。
    """
    if config is None:
        from config import get_default_config
        config = get_default_config()

    eq = config.env_quality
    si = config.sleep_impact
    cp = config.control_policy

    models = {
        "env_quality_classifier": EnvQualityClassifier(
            numeric_dim=eq.numeric_dim,
            odor_vocab_size=eq.odor_vocab_size,
            odor_emb_dim=eq.odor_emb_dim,
            hidden_dim=eq.hidden_dim,
            risk_dim=eq.risk_dim,
            num_classes=eq.num_classes,
        ),
        "sleep_impact_predictor": SleepImpactPredictor(
            env_seq_dim=si.env_seq_dim,
            static_dim=si.static_dim,
            hist_dim=si.hist_dim,
            lstm_hidden_dim=si.lstm_hidden_dim,
            lstm_layers=si.lstm_layers,
            fusion_hidden_dim=si.fusion_hidden_dim,
            output_dim=si.output_dim,
            dropout=si.dropout,
        ),
        "control_policy_model": ControlPolicyModel(
            state_dim=cp.state_dim,
            discrete_action_dim=cp.discrete_action_dim,
            continuous_action_dim=cp.continuous_action_dim,
            hidden_dim=cp.hidden_dim,
            rnn_layers=cp.rnn_layers,
            rnn_type=cp.rnn_type,
            action_log_std_init=cp.action_log_std_init,
        ),
    }
    return models