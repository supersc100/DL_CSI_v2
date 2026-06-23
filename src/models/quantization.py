"""Scalar quantization utilities for Phase 2 limited-feedback experiments."""
import torch
import torch.nn as nn


class ScalarQuantizer(nn.Module):
    """Simulate per-sample scalar quantization of complex tensors.

    Real and imaginary parts are quantized separately using a uniform quantizer
    with a dynamic range computed per sample.  This mimics adaptive scalar
    quantization and can be used for quantization-aware training (QAT).

    Args:
        num_bits: number of bits per real value (total 2*num_bits per complex
            sample).
    """

    def __init__(self, num_bits: int = 16):
        super().__init__()
        self.num_bits = int(num_bits)
        self.num_levels = 2 ** self.num_bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize/dequantize a complex tensor.

        Args:
            x: complex tensor of any shape.

        Returns:
            Quantized-dequantized complex tensor of the same shape.
        """
        if not torch.is_complex(x):
            raise TypeError(f"ScalarQuantizer expects complex input, got {x.dtype}")

        real_q = self._quantize_dequantize(x.real)
        imag_q = self._quantize_dequantize(x.imag)
        return torch.complex(real_q, imag_q)

    def _quantize_dequantize(self, t: torch.Tensor) -> torch.Tensor:
        """Quantize and dequantize a real tensor with per-sample min/max."""
        # Flatten all dims except batch (assume first dim is batch).
        if t.ndim == 0:
            # Scalar tensor: fall back to no quantization.
            return t
        original_shape = t.shape
        t_flat = t.view(t.shape[0], -1)

        t_min = t_flat.min(dim=1, keepdim=True)[0]
        t_max = t_flat.max(dim=1, keepdim=True)[0]
        scale = (t_max - t_min) / (self.num_levels - 1)
        scale = scale.clamp_min(1e-8)

        t_int = torch.round((t_flat - t_min) / scale).clamp(0, self.num_levels - 1)
        t_dq = t_int * scale + t_min
        return t_dq.view(original_shape)


class IdentityQuantizer(nn.Module):
    """No-op quantizer for convenience (passes input through unchanged)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x
