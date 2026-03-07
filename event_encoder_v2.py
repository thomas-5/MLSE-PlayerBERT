from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventTokenEncoder(nn.Module):
    """Encode flattened event attributes into event tokens."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        numeric_dim: int,
        d_model: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = list(vocab_sizes.keys())
        self.safe_names = [f"f{i}" for i in range(len(self.features))]
        self.name_map = dict(zip(self.features, self.safe_names))

        self.value_embeds = nn.ModuleDict(
            {
                self.name_map[f]: nn.Embedding(vocab_sizes[f], d_model)
                for f in self.features
            }
        )
        self.feature_embeds = nn.Embedding(len(self.features), d_model)
        self.numeric_proj = nn.Sequential(
            nn.Linear(numeric_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "feature_idx",
            torch.arange(len(self.features), dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        feat_ids: torch.Tensor,
        numeric_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # feat_ids: (B, F), numeric_values: (B, N)
        batch_size, num_features = feat_ids.shape
        if num_features != len(self.features):
            raise ValueError(
                f"Expected {len(self.features)} features, got {num_features}"
            )

        value_tokens = []
        for i, feat in enumerate(self.features):
            safe_name = self.name_map[feat]
            value_tokens.append(self.value_embeds[safe_name](feat_ids[:, i]))
        value_tokens = torch.stack(value_tokens, dim=1)  # (B, F, D)

        feat_emb = self.feature_embeds(self.feature_idx.to(feat_ids.device))
        feat_tokens = value_tokens + feat_emb.unsqueeze(0)

        numeric_token = self.numeric_proj(numeric_values).unsqueeze(1)  # (B, 1, D)
        event_tokens = torch.cat([numeric_token, feat_tokens], dim=1)
        event_tokens = self.dropout(event_tokens)

        event_mask = torch.ones(
            event_tokens.shape[0],
            event_tokens.shape[1],
            dtype=torch.bool,
            device=event_tokens.device,
        )
        return event_tokens, event_mask


class RelationalSelfAttention(nn.Module):
    """Self-attention over player tokens with geometric relational bias."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        hidden = max(16, d_model // 8)
        self.rel_bias = nn.Sequential(
            nn.Linear(4, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.dropout = nn.Dropout(dropout)

    def _pairwise_bias(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: (B, P, 2), typically relative dx/dy
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, P, P, 2)
        dist = torch.linalg.norm(diff, dim=-1)
        angle = torch.atan2(diff[..., 1], diff[..., 0] + 1e-8)
        rel = torch.stack(
            [
                dist,
                torch.sin(angle),
                torch.cos(angle),
                diff[..., 0] * diff[..., 1],
            ],
            dim=-1,
        )
        return self.rel_bias(rel).squeeze(-1)  # (B, P, P)

    def forward(
        self,
        x: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, P, D), coords: (B, P, 2), mask: (B, P)
        batch_size, num_tokens, _ = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, num_tokens, 3, self.num_heads, self.head_dim
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, P, Hd)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, P, P)
        scores = scores + self._pairwise_bias(coords).unsqueeze(1)

        key_mask = mask[:, None, None, :]
        scores = scores.masked_fill(~key_mask, -1e4)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, P, Hd)
        query_mask = mask[:, None, :, None]
        out = out * query_mask

        out = out.transpose(1, 2).contiguous().view(batch_size, num_tokens, self.d_model)
        return self.out_proj(out)


class RelationalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = RelationalSelfAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, coords, mask)
        x = x * mask.unsqueeze(-1)
        x = x + self.ff(self.norm2(x))
        x = x * mask.unsqueeze(-1)
        return x


class FrameGraphEncoder(nn.Module):
    """Encode freeze-frame players as a relational token set."""

    def __init__(
        self,
        player_dim: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(player_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.layers = nn.ModuleList(
            [
                RelationalTransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        player_tokens: torch.Tensor,
        player_mask: torch.Tensor,
        mask_positions: Optional[torch.Tensor] = None,
        mask_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # player_tokens: (B, P, player_dim), player_mask: (B, P)
        x = self.input_proj(player_tokens)
        if mask_positions is not None and mask_token is not None:
            x = torch.where(mask_positions.unsqueeze(-1), mask_token.view(1, 1, -1), x)
        x = self.dropout(x)
        x = x * player_mask.unsqueeze(-1)

        coords = player_tokens[..., :2]
        for layer in self.layers:
            x = layer(x, coords, player_mask)
        return x


class EventConditionedSceneEncoder(nn.Module):
    """V2 encoder: event tokens + relational 360 scene + cross-attention fusion."""

    def __init__(
        self,
        vocab_sizes: Dict[str, int],
        numeric_dim: int,
        player_dim: int = 8,
        d_model: int = 192,
        out_dim: int = 128,
        event_layers: int = 2,
        frame_layers: int = 2,
        num_heads: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.features = list(vocab_sizes.keys())
        self.d_model = d_model
        self.out_dim = out_dim

        self.event_tokens = EventTokenEncoder(
            vocab_sizes=vocab_sizes,
            numeric_dim=numeric_dim,
            d_model=d_model,
            dropout=dropout,
        )
        self.event_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        event_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.event_encoder = nn.TransformerEncoder(event_layer, num_layers=event_layers)

        self.frame_encoder = FrameGraphEncoder(
            player_dim=player_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=frame_layers,
            dropout=dropout,
        )
        self.frame_mask_token = nn.Parameter(torch.zeros(d_model))

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, out_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        feat_ids: torch.Tensor,
        numeric_values: torch.Tensor,
        frame_tokens: torch.Tensor,
        frame_mask: torch.Tensor,
        frame_mask_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = feat_ids.shape[0]

        event_tokens, event_mask = self.event_tokens(feat_ids, numeric_values)
        ev_query = self.event_query.expand(batch_size, -1, -1)
        event_input = torch.cat([ev_query, event_tokens], dim=1)
        event_input_mask = torch.cat(
            [
                torch.ones(batch_size, 1, dtype=torch.bool, device=feat_ids.device),
                event_mask,
            ],
            dim=1,
        )
        event_hidden = self.event_encoder(
            event_input,
            src_key_padding_mask=~event_input_mask,
        )

        frame_hidden = self.frame_encoder(
            player_tokens=frame_tokens,
            player_mask=frame_mask,
            mask_positions=frame_mask_positions,
            mask_token=self.frame_mask_token,
        )

        # Event query attends over both encoded event tokens and frame tokens.
        joint_tokens = torch.cat([event_hidden[:, 1:, :], frame_hidden], dim=1)
        joint_mask = torch.cat([event_input_mask[:, 1:], frame_mask], dim=1)
        query = event_hidden[:, :1, :]
        cross_out, _ = self.cross_attn(
            query=query,
            key=joint_tokens,
            value=joint_tokens,
            key_padding_mask=~joint_mask,
            need_weights=False,
        )

        fused = self.final_norm(query + cross_out).squeeze(1)
        embedding = self.out_proj(fused)
        projection = F.normalize(self.proj_head(embedding), dim=-1)

        return {
            "embedding": embedding,  # E_i
            "projection": projection,  # for contrastive loss
            "feature_tokens": event_hidden[:, 2:, :],  # skip [EV] and numeric token
            "frame_tokens": frame_hidden,
            "query_token": fused,
        }


class MaskedFeatureHead(nn.Module):
    """Per-feature classifiers for masked attribute modeling."""

    def __init__(self, vocab_sizes: Dict[str, int], d_model: int) -> None:
        super().__init__()
        self.features = list(vocab_sizes.keys())
        self.safe_names = [f"f{i}" for i in range(len(self.features))]
        self.name_map = dict(zip(self.features, self.safe_names))
        self.heads = nn.ModuleDict(
            {
                self.name_map[f]: nn.Linear(d_model, vocab_sizes[f])
                for f in self.features
            }
        )

    def forward(self, feature_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = {}
        for i, feature in enumerate(self.features):
            logits[feature] = self.heads[self.name_map[feature]](feature_tokens[:, i, :])
        return logits


class FrameReconstructionHead(nn.Module):
    """Reconstruct masked player tokens in freeze-frame."""

    def __init__(self, d_model: int, player_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, player_dim),
        )

    def forward(self, frame_tokens: torch.Tensor) -> torch.Tensor:
        return self.net(frame_tokens)


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """Symmetric InfoNCE on two stochastic views."""
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    batch_size = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)  # (2B, D)

    sim = torch.matmul(z, z.transpose(0, 1)) / temperature
    sim = sim.masked_fill(
        torch.eye(2 * batch_size, device=z.device, dtype=torch.bool),
        -1e9,
    )

    targets = torch.arange(batch_size, device=z.device)
    targets = torch.cat([targets + batch_size, targets], dim=0)
    return F.cross_entropy(sim, targets)
