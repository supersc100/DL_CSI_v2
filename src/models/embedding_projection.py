"""Continuous feature projection that replaces the LLM token embedding."""
import torch
import torch.nn as nn


class ContinuousEmbeddingProjection(nn.Module):
    """Project local feature vectors into the LLM hidden space.

    Instead of token IDs, we feed the LLM `inputs_embeds` directly. This layer
    maps each soft token (CSI feature, temporal feature, env feature) from
    `feature_dim` to `llm_hidden_dim`, producing continuous embeddings.

    Input shape:  [B, num_virtual_tokens, feature_dim]
    Output shape: [B, num_virtual_tokens, llm_hidden_dim]
    """

    def __init__(
        self,
        feature_dim: int,
        llm_hidden_dim: int,
        num_virtual_tokens: int = 3,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.llm_hidden_dim = llm_hidden_dim
        self.num_virtual_tokens = num_virtual_tokens

        self.proj = nn.Linear(feature_dim, llm_hidden_dim)
        self.norm = nn.LayerNorm(llm_hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x)
        return x
