import math
import torch
import torch.nn as nn


def cyclical_enc(v: torch.Tensor, period: float, k_max: int) -> torch.Tensor:
    """
    v: [B,T], with values in [0, period) or any real values (wrapped modulo period).
    Returns: [B,T,2*k_max]
    """
    if k_max <= 0:
        raise ValueError("k_max must be > 0")
    # Wrap values modulo the period to keep them within bounds.
    v = torch.remainder(v, period)
    v01 = v / period  # [0,1)
    ks = torch.arange(1, k_max + 1, device=v.device, dtype=v.dtype)  # [K]
    phase = 2.0 * math.pi * v01.unsqueeze(-1) * ks  # [B,T,K]
    return torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)



class TemporalEncoder2(nn.Module):
    """Encode TAANP's minute, hour and weekday columns into time features."""

    def __init__(self, d_phi: int, hidden: int = 128):
        super().__init__()
        # Retain these inactive buffers for strict loading of existing checkpoints.
        self.register_buffer("t0", torch.tensor(0.0), persistent=True)
        self.register_buffer("t_scale", torch.tensor(1.0), persistent=True)

        # Two minute, three hour and two weekday harmonics, each with sin/cos.
        self.mlp = nn.Sequential(
            nn.Linear(14, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(hidden, d_phi),
        )

    def forward(self, x_time: torch.Tensor) -> torch.Tensor:
        """Map [B, T, time_columns] to [B, T, d_phi]."""
        assert x_time.dim() == 3, f"expect [B,T,T_in], got {tuple(x_time.shape)}"
        time_of_hour = x_time[:, :, 0]
        # Preserve the original minute/second detection used for this column.
        period = 3600.0 if float(time_of_hour.max().detach().cpu().item()) > 100.0 else 60.0
        feats = [
            cyclical_enc(time_of_hour, period=period, k_max=2),
            cyclical_enc(x_time[:, :, 1], period=24.0, k_max=3),
            cyclical_enc(x_time[:, :, 2], period=7.0, k_max=2),
        ]
        return self.mlp(torch.cat(feats, dim=-1))
