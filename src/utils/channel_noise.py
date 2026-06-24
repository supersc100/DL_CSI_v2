"""Additive white Gaussian noise (AWGN) injection for SNR sweeps.

Used by the simulation harness to evaluate trained models under different
receive SNRs without regenerating data.  Noise is injected on the *measured*
CSI (uplink input and downlink sparse pilots); the supervision target stays
clean.
"""
from typing import Optional

import torch


def add_awgn(
    x: torch.Tensor,
    snr_db: float,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Add complex AWGN to a complex tensor at a given SNR.

    SNR is defined per-tensor from the average signal power::

        P_sig    = mean(|x|^2)
        noise_var = P_sig / 10^(snr_db / 10)

    The real and imaginary parts each receive ``N(0, noise_var / 2)`` so that
    the total complex noise power equals ``noise_var``.

    Args:
        x: complex tensor of any shape (one sample, or a batch).
        snr_db: target signal-to-noise ratio in dB.
        generator: optional ``torch.Generator`` for reproducible noise.  Pass a
            per-sample seeded generator so the same sample gets a consistent
            noise realization across different SNR points.

    Returns:
        Complex tensor of the same shape/dtype with AWGN added.
    """
    if not torch.is_complex(x):
        raise TypeError(f"add_awgn expects a complex tensor, got {x.dtype}")

    p_sig = x.abs().square().mean()
    # Guard against all-zero tensors (e.g. masked-out pilots before masking).
    p_sig = p_sig.clamp_min(1e-20)
    noise_var = p_sig / (10.0 ** (snr_db / 10.0))
    std = torch.sqrt(noise_var / 2.0)

    noise_real = torch.empty_like(x.real).normal_(mean=0.0, std=1.0, generator=generator)
    noise_imag = torch.empty_like(x.imag).normal_(mean=0.0, std=1.0, generator=generator)
    noise = torch.complex(noise_real, noise_imag) * std.to(x.real.dtype)
    return x + noise
