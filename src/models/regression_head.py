"""Regression head mapping fused features to predicted downlink CSI."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RegressionHead(nn.Module):
    """Predict downlink CSI in angle-delay domain from fused features.

    Output modes:
        - "ri":      predict real and imaginary parts, then reconstruct complex.
        - "amp_phase": predict amplitude and phase (stable phase via tanh + pi).

    When ``use_residual_log_amp`` is true, the amplitude branch predicts a
    residual log-amplitude correction on top of the current UL magnitude:

        log|H_DL_pred| = log|H_UL| + delta_log_amp

    This guarantees that the Stage1 output is explicitly conditioned on the
    uplink CSI and cannot collapse to a fixed downlink template.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        output_mode: str = "ri",
        dropout: float = 0.1,
        use_residual_log_amp: bool = False,
        residual_eps: float = 1e-6,
        max_log_ratio: float = 5.0,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.output_dim = output_dim
        self.use_residual_log_amp = use_residual_log_amp
        self.residual_eps = residual_eps
        self.max_log_ratio = max_log_ratio

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        target_shape: tuple,
        ul_log_mag: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # hidden: [B, input_dim]
        # Cast to float32 so complex conversion is well-defined and loss is stable.
        x = self.mlp(hidden.to(torch.float32))

        if self.output_mode == "ri":
            # output_dim = 2 * num_elements. Split into real/imag.
            num_elements = self.output_dim // 2
            real = x[:, :num_elements]
            imag = x[:, num_elements:]
            pred = torch.view_as_complex(torch.stack([real, imag], dim=-1))
        elif self.output_mode == "amp_phase":
            num_elements = self.output_dim // 2
            if self.use_residual_log_amp:
                if ul_log_mag is None:
                    raise ValueError(
                        "use_residual_log_amp=True requires ul_log_mag to be passed to the head."
                    )
                # Residual log-amplitude: log|H_DL| = log|H_UL| + delta.
                # Bound delta to avoid exp() overflow/underflow while still
                # allowing a wide dynamic range (±max_log_ratio nats).
                delta_log_amp = torch.tanh(x[:, :num_elements]) * self.max_log_ratio
                phase_raw = x[:, num_elements:]
                amp = torch.exp(ul_log_mag + delta_log_amp)
                phase = torch.tanh(phase_raw) * math.pi
            else:
                amp = F.softplus(x[:, :num_elements]) + 1e-6
                phase = torch.tanh(x[:, num_elements:]) * math.pi
            pred = amp * torch.exp(1j * phase)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        pred = pred.view(target_shape)
        return pred
