import torch
import torch.nn as nn


class MLPHead(nn.Module):
    """Two-layer MLP regression head for gene expression prediction.
    
    Same architecture is used for all encoders to ensure fair comparison.
    Only the input dimension changes to match each encoder's embedding size.
    """

    def __init__(self, input_dim: int, output_dim: int = 50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def get_features(encoder, patches: torch.Tensor, encoder_name: str) -> torch.Tensor:
    """Unified feature extraction for HuggingFace and timm encoder APIs.
    
    DINOv3 returns a BaseModelOutput; we take the CLS token.
    UNI (timm) returns embeddings directly.
    """
    if encoder_name == 'dinov3':
        return encoder(pixel_values=patches).last_hidden_state[:, 0, :]
    return encoder(patches)
