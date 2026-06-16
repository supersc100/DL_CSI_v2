"""Regression head mapping LLM hidden states to predicted downlink CSI."""
import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Predict downlink CSI in angle-delay domain from LLM last hidden state.

    Output modes:
        - "ri":      predict real and imaginary parts, then reconstruct complex.
        - "amp_phase": predict amplitude and phase (stable phase via tanh + pi).
    """

    def __init__(
        self,
        llm_hidden_dim: int,
        output_dim: int,
        hidden_dim: int = 1024,
        output_mode: str = "ri",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.output_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(llm_hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, hidden: torch.Tensor, target_shape: tuple) -> torch.Tensor:
        # hidden: [B, llm_hidden_dim]
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
            amp = torch.nn.functional.softplus(x[:, :num_elements]) + 1e-6
            phase = torch.tanh(x[:, num_elements:]) * 3.141592653589793
            pred = amp * torch.exp(1j * phase)
        else:
            raise ValueError(f"Unknown output_mode: {self.output_mode}")

        pred = pred.view(target_shape)
        return pred
