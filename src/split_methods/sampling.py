"""Two-phase exact rejection sampling for serial split-device speculation.

Only sampled q scalars cross in phase one. Phase two requests one complete q
row for each rejected request, not the whole [batch, gamma, vocabulary] cube.
Reference torch operations intentionally precede graph/kernel optimization.
"""

from dataclasses import dataclass
import torch


@dataclass
class AcceptanceDecision:
    lengths: torch.Tensor
    rejected_rows: torch.Tensor
    rejection_positions: torch.Tensor
    target_rows: torch.Tensor


def decide_acceptance(*, candidates, sampled_q, target_probs, uniforms):
    batch, gamma = candidates.shape
    if sampled_q.shape != (batch, gamma) or uniforms.shape != (batch, gamma):
        raise ValueError("sampled q and uniforms must match the seven proposal positions")
    if target_probs.shape[:2] != (batch, gamma + 1):
        raise ValueError("target must verify proposal width plus one bonus row")
    selected_p = target_probs[:, :gamma].gather(2, candidates[..., None]).squeeze(-1)
    accept = uniforms * sampled_q < selected_p
    lengths = accept.to(torch.int64).cumprod(1).sum(1)
    rows = torch.arange(batch, device=candidates.device)
    rejected_rows = rows[lengths < gamma]
    return AcceptanceDecision(
        lengths=lengths,
        rejected_rows=rejected_rows,
        rejection_positions=lengths[rejected_rows],
        target_rows=target_probs[rows, lengths],
    )


def finish_sampling(decision, rejected_q_rows, *, generator=None, uniforms=None):
    target = decision.target_rows
    expected = (decision.rejected_rows.numel(), target.shape[-1])
    if rejected_q_rows.shape != expected:
        raise ValueError(f"expected rejected q shape {expected}, got {rejected_q_rows.shape}")
    probs = target.clone()
    if decision.rejected_rows.numel():
        residual = (target[decision.rejected_rows] - rejected_q_rows).clamp_min(0)
        mass = residual.sum(-1, keepdim=True)
        # Numerical cancellation fallback matches the colocated verifier.
        probs[decision.rejected_rows] = torch.where(
            mass > 1e-8,
            residual / mass.clamp_min(1e-30),
            target[decision.rejected_rows],
        )
    if uniforms is None:
        return torch.multinomial(probs, 1, generator=generator).squeeze(-1)
    if uniforms.shape != (target.shape[0],):
        raise ValueError("one final uniform is required per request")
    return (uniforms[:, None] >= probs.cumsum(-1)).sum(-1).clamp_max(probs.shape[-1] - 1)
