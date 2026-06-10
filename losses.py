import torch
import torch.nn.functional as F


def env_quality_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    comfort_targets: torch.Tensor | None = None,
    class_weight: float = 1.0,
    risk_weight: float = 1.0,
    comfort_weight: float = 1.0,
) -> torch.Tensor:
    """多任务环境质量损失。

    risk_logits 与 class_logits 共享相同标签（0=无风险/舒适,1=过热,2=过冷,3=气味不适），
    均使用交叉熵损失。risk_head 与 class_head 从共享表征中学习不同视角。
    """
    loss = class_weight * F.cross_entropy(outputs["class_logits"], labels)

    if "risk_logits" in outputs:
        loss = loss + risk_weight * F.cross_entropy(outputs["risk_logits"], labels)

    if comfort_targets is not None and "comfort_score" in outputs:
        loss = loss + comfort_weight * F.mse_loss(
            outputs["comfort_score"].unsqueeze(-1), comfort_targets
        )

    return loss


def sleep_impact_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    output_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """睡眠影响预测的回归损失。"""
    loss = F.smooth_l1_loss(outputs, targets, reduction="none")
    if output_weights is not None:
        if output_weights.dim() == 1:
            output_weights = output_weights.unsqueeze(0)
        loss = loss * output_weights
    return loss.mean()


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> torch.Tensor:
    """计算广义优势估计 (Generalized Advantage Estimation)。

    沿时间步方向（dim=1）反向递推计算 GAE。

    Args:
        rewards:    (batch, rollout_len)
        values:     (batch, rollout_len)  或 (batch, rollout_len + 1) 若提供 next_value
        dones:      (batch, rollout_len)  1 表示终止（或截断）
        gamma:      折扣因子
        lam:        GAE λ

    Returns:
        advantages: (batch, rollout_len)
        returns:    (batch, rollout_len)  advantages + values
    """
    batch_size, rollout_len = rewards.shape

    # values 应为 (batch, rollout_len)，若为 (batch, rollout_len+1) 则取前 rollout_len 个
    if values.shape[1] == rollout_len + 1:
        next_values = values[:, 1:]
        values = values[:, :-1]
    else:
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:]

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(batch_size, device=rewards.device)

    for t in reversed(range(rollout_len)):
        # TD 误差: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        delta = rewards[:, t] + gamma * next_values[:, t] * (1 - dones[:, t]) - values[:, t]
        # GAE: A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
        last_advantage = delta + gamma * lam * (1 - dones[:, t]) * last_advantage
        advantages[:, t] = last_advantage

    returns = advantages + values
    return advantages, returns


def ppo_loss(
    eval_out: dict[str, torch.Tensor],
    old_log_prob: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    discrete_actions: torch.Tensor,
    continuous_actions: torch.Tensor,
    clip_ratio: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict[str, torch.Tensor]:
    """PPO Clipped 损失函数（支持离散 + 连续混合动作空间）。

    Args:
        eval_out:           模型 evaluate_actions() 的输出，
                            包含 "log_prob", "state_value", "discrete_logits",
                            "continuous_mean", "continuous_log_std", "entropy"
        old_log_prob:       (batch,)  采样时的 log_prob
        advantages:         (batch,)  经 GAE 计算的 advantage，需要已标准化
        returns:            (batch,)  GAE returns
        discrete_actions:   (batch,)  离散动作标签
        continuous_actions: (batch, cont_dim)  连续动作值
        clip_ratio:         PPO clipping ε
        value_coef:         价值损失系数
        entropy_coef:       熵奖励系数
        max_grad_norm:      梯度裁剪阈值

    Returns:
        dict: 包含 loss, policy_loss, value_loss, entropy, approx_kl,
              clip_fraction 的字典。
    """
    # ── Policy loss (clipped) ──
    log_prob = eval_out["log_prob"]  # (batch,)
    ratio = torch.exp(log_prob - old_log_prob)

    # PPO clipped objective
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()

    # ── Value loss ──
    state_value = eval_out["state_value"]  # (batch,)
    # Clip value 也做约束（标准 PPO 做法）
    value_pred_clipped = old_log_prob.detach().new_zeros(state_value.shape)  # 占位，我们不 clip value 预测
    # 简化：仅使用 MSE loss（不需要 value clipping 简化版）
    value_loss = F.mse_loss(state_value, returns)

    # ── Entropy bonus ──
    entropy = eval_out.get("entropy", torch.tensor(0.0, device=advantages.device))
    if entropy.dim() == 0:
        entropy_mean = entropy
    else:
        entropy_mean = entropy.mean()

    # ── 总损失 ──
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_mean

    # ── 诊断统计 ──
    with torch.no_grad():
        approx_kl = ((ratio - 1) - (log_prob - old_log_prob)).mean()
        clip_fraction = ((ratio - 1).abs() > clip_ratio).float().mean()

    return {
        "loss": loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "approx_kl": approx_kl,
        "clip_fraction": clip_fraction,
    }


