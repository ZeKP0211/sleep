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


def control_policy_losses(
    eval_out: dict[str, torch.Tensor],
    action_discrete: torch.Tensor,
    action_cont: torch.Tensor,
    discrete_weight: float = 1.0,
    continuous_weight: float = 1.0,
    entropy_coef: float = 0.0,
) -> dict[str, torch.Tensor]:
    """控制策略的监督式模仿损失。"""
    discrete_loss = F.cross_entropy(eval_out["discrete_logits"], action_discrete)
    continuous_loss = F.mse_loss(eval_out["continuous_mean"], action_cont)
    loss = discrete_weight * discrete_loss + continuous_weight * continuous_loss

    entropy = torch.tensor(0.0, device=loss.device)
    if entropy_coef != 0.0 and "continuous_log_std" in eval_out:
        # 离散动作熵
        discrete_dist = torch.distributions.Categorical(logits=eval_out["discrete_logits"])
        discrete_entropy = discrete_dist.entropy().mean()

        # 连续动作熵
        continuous_std = torch.exp(eval_out["continuous_log_std"])
        continuous_dist = torch.distributions.Normal(eval_out["continuous_mean"], continuous_std)
        continuous_entropy = continuous_dist.entropy().sum(dim=-1).mean()

        entropy = discrete_entropy + continuous_entropy
        loss = loss - entropy_coef * entropy

    return {
        "loss": loss,
        "discrete_loss": discrete_loss,
        "continuous_loss": continuous_loss,
        "entropy": entropy,
    }
