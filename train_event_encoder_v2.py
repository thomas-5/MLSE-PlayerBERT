#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from event_encoder_v2 import (
    EventConditionedSceneEncoder,
    FrameReconstructionHead,
    MaskedFeatureHead,
    nt_xent_loss,
)

try:
    from tqdm import tqdm
except Exception:  # noqa: BLE001
    def tqdm(iterable=None, **kwargs):  # type: ignore[override]
        return iterable if iterable is not None else []


UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

EXCLUDE_PREFIXES = [
    "id",
    "event_uuid",
    "related_events",
    "freeze_frame",
    "visible_area",
]
EXCLUDE_CONTAINS = [
    "location",
    "end_location",
    "angle",
    "length",
    "statsbomb_xg",
]
INCLUDE_SUBSTRINGS = ["bucket"]


def iter_json_objects(path: Path) -> Iterable[dict]:
    """Yield JSON objects from either JSONL or concatenated-JSON files.

    Supports:
    - Proper JSONL (one object per line)
    - Concatenated objects with no newline separator: {...}{...}{...}
    """
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(1 << 20)  # 1 MB
            if not chunk:
                break
            buf += chunk

            while True:
                buf = buf.lstrip()
                if not buf:
                    break
                try:
                    obj, idx = decoder.raw_decode(buf)
                except json.JSONDecodeError:
                    # Need more bytes for a complete object.
                    break
                yield obj
                buf = buf[idx:]

        # Parse any remaining trailing object.
        buf = buf.lstrip()
        if buf:
            obj, idx = decoder.raw_decode(buf)
            if buf[idx:].strip():
                raise json.JSONDecodeError(
                    "Trailing non-JSON content",
                    buf,
                    idx,
                )
            yield obj


def normalize_value(val) -> str:
    if val is None:
        return UNK_TOKEN
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, (dict, list)):
        return UNK_TOKEN
    return str(val)


def should_use_feature(key: str) -> bool:
    if any(key.startswith(prefix) for prefix in EXCLUDE_PREFIXES):
        return False
    if any(tok in key for tok in EXCLUDE_CONTAINS):
        if not any(inc in key for inc in INCLUDE_SUBSTRINGS):
            return False
    return True


def resolve_default_data_path() -> Path:
    candidates = [
        Path("open-data/data/processed/events360_v4.jsonl"),
        Path("open-data/data/processed/events360_v2.jsonl"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Could not find default dataset path. Pass --data_path explicitly."
    )


def derive_event_features(path: Path, max_rows: int | None) -> List[str]:
    cols = set()
    for i, ev in enumerate(iter_json_objects(path)):
        cols.update(ev.keys())
        if max_rows is not None and (i + 1) >= max_rows:
            break
    features = [key for key in sorted(cols) if should_use_feature(key)]
    return features


def build_feature_vocab(
    path: Path,
    features: List[str],
    min_freq: int,
    max_rows: int | None,
) -> Dict[str, Dict[str, int]]:
    counts = {feature: Counter() for feature in features}
    for i, ev in enumerate(iter_json_objects(path)):
        for feature in features:
            counts[feature][normalize_value(ev.get(feature))] += 1
        if max_rows is not None and (i + 1) >= max_rows:
            break

    vocab: Dict[str, Dict[str, int]] = {}
    for feature in features:
        vocab[feature] = {UNK_TOKEN: 0, MASK_TOKEN: 1}
        for value, count in counts[feature].most_common():
            if value in vocab[feature]:
                continue
            if count < min_freq:
                continue
            vocab[feature][value] = len(vocab[feature])
    return vocab


def extract_numeric(ev: dict) -> torch.Tensor:
    loc = ev.get("location")
    if isinstance(loc, list) and len(loc) >= 2:
        x = float(loc[0]) / 120.0
        y = float(loc[1]) / 80.0
    else:
        x, y = 0.0, 0.0

    minute = float(ev.get("minute") or 0.0) / 120.0
    second = float(ev.get("second") or 0.0) / 60.0
    period = float(ev.get("period") or 0.0) / 5.0
    duration = float(ev.get("duration") or 0.0)
    under_pressure = 1.0 if bool(ev.get("under_pressure")) else 0.0
    counterpress = 1.0 if bool(ev.get("counterpress")) else 0.0
    return torch.tensor(
        [
            x,
            y,
            minute,
            second,
            period,
            math.log1p(max(0.0, duration)),
            under_pressure,
            counterpress,
        ],
        dtype=torch.float32,
    )


def extract_freeze_frame(ev: dict, max_players: int | None) -> tuple[torch.Tensor, int]:
    loc = ev.get("location")
    ax = float(loc[0]) if isinstance(loc, list) and len(loc) >= 2 else 0.0
    ay = float(loc[1]) if isinstance(loc, list) and len(loc) >= 2 else 0.0

    players = []
    freeze_frame = ev.get("freeze_frame") or []
    if isinstance(freeze_frame, list):
        for p in freeze_frame:
            if not isinstance(p, dict):
                continue
            ploc = p.get("location")
            if not isinstance(ploc, list) or len(ploc) < 2:
                continue
            dx = (float(ploc[0]) - ax) / 120.0
            dy = (float(ploc[1]) - ay) / 80.0
            dist = math.sqrt(dx * dx + dy * dy)
            angle = math.atan2(dy, dx + 1e-8)
            players.append(
                [
                    dx,
                    dy,
                    dist,
                    math.sin(angle),
                    math.cos(angle),
                    1.0 if bool(p.get("teammate")) else 0.0,
                    1.0 if bool(p.get("keeper")) else 0.0,
                    1.0 if bool(p.get("actor")) else 0.0,
                ]
            )

    valid_count = len(players)
    if max_players is not None:
        players = players[:max_players]
        valid_count = min(valid_count, max_players)

    if not players:
        players = [[0.0] * 8]
    return torch.tensor(players, dtype=torch.float32), valid_count


class Events360Dataset(Dataset):
    def __init__(
        self,
        path: Path,
        features: List[str],
        feature_vocab: Dict[str, Dict[str, int]],
        max_rows: int | None = None,
        max_players: int | None = 22,
    ) -> None:
        self.events = []
        for i, ev in enumerate(iter_json_objects(path)):
            self.events.append(ev)
            if max_rows is not None and (i + 1) >= max_rows:
                break
        self.features = features
        self.feature_vocab = feature_vocab
        self.max_players = max_players

    def __len__(self) -> int:
        return len(self.events)

    def __getitem__(self, idx: int) -> dict:
        ev = self.events[idx]
        feat_ids = []
        for feature in self.features:
            val = normalize_value(ev.get(feature))
            feat_ids.append(self.feature_vocab[feature].get(val, 0))
        feat_ids_t = torch.tensor(feat_ids, dtype=torch.long)
        numeric_t = extract_numeric(ev)
        frame_t, frame_valid_count = extract_freeze_frame(ev, self.max_players)
        return {
            "feat_ids": feat_ids_t,
            "numeric": numeric_t,
            "frame_tokens": frame_t,
            "frame_valid_count": frame_valid_count,
        }


def collate_batch(batch: List[dict]) -> dict:
    feat_ids = torch.stack([row["feat_ids"] for row in batch], dim=0)
    numeric = torch.stack([row["numeric"] for row in batch], dim=0)

    frame_dim = batch[0]["frame_tokens"].shape[-1]
    max_players = max(row["frame_tokens"].shape[0] for row in batch)
    frames = torch.zeros(len(batch), max_players, frame_dim, dtype=torch.float32)
    frame_mask = torch.zeros(len(batch), max_players, dtype=torch.bool)
    for i, row in enumerate(batch):
        players = row["frame_tokens"]
        frames[i, : players.shape[0]] = players
        valid_count = min(row["frame_valid_count"], players.shape[0])
        if valid_count > 0:
            frame_mask[i, :valid_count] = True

    return {
        "feat_ids": feat_ids,
        "numeric": numeric,
        "frame_tokens": frames,
        "frame_mask": frame_mask,
    }


def mask_feature_ids(
    feat_ids: torch.Tensor,
    feature_vocab_sizes: List[int],
    mask_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked = feat_ids.clone()
    labels = torch.full_like(feat_ids, fill_value=-100)

    batch_size, num_features = feat_ids.shape
    mask_positions = torch.rand(batch_size, num_features, device=feat_ids.device) < mask_prob
    labels[mask_positions] = feat_ids[mask_positions]

    for feature_idx, vocab_size in enumerate(feature_vocab_sizes):
        pos = mask_positions[:, feature_idx]
        if not pos.any():
            continue
        row_idx = torch.nonzero(pos, as_tuple=False).squeeze(1)
        draws = torch.rand(row_idx.shape[0], device=feat_ids.device)
        mask_choice = draws < 0.8
        rand_choice = (draws >= 0.8) & (draws < 0.9)

        if mask_choice.any():
            masked[row_idx[mask_choice], feature_idx] = 1  # [MASK]
        if rand_choice.any():
            masked[row_idx[rand_choice], feature_idx] = torch.randint(
                low=0,
                high=vocab_size,
                size=(int(rand_choice.sum().item()),),
                device=feat_ids.device,
            )

    return masked, labels


def mask_frame_positions(frame_mask: torch.Tensor, mask_prob: float) -> torch.Tensor:
    sampled = torch.rand_like(frame_mask.float()) < mask_prob
    return sampled & frame_mask


def jitter_frame_tokens(
    frame_tokens: torch.Tensor,
    frame_mask: torch.Tensor,
    noise_std: float = 0.01,
) -> torch.Tensor:
    jittered = frame_tokens.clone()
    noise = torch.randn_like(jittered[:, :, :2]) * noise_std
    jittered[:, :, :2] = jittered[:, :, :2] + noise * frame_mask.unsqueeze(-1)
    return jittered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V2 event-conditioned scene encoder.")
    parser.add_argument("--data_path", type=Path, default=None, help="Flattened JSONL events path")
    parser.add_argument("--output", type=Path, default=Path("models/event_encoder_v2.pt"))
    parser.add_argument("--max_rows", type=int, default=None, help="For quick iteration/debugging")
    parser.add_argument("--max_players", type=int, default=22)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--out_dim", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--event_layers", type=int, default=2)
    parser.add_argument("--frame_layers", type=int, default=2)
    parser.add_argument("--mask_prob_features", type=float, default=0.15)
    parser.add_argument("--mask_prob_players", type=float, default=0.20)
    parser.add_argument("--w_mam", type=float, default=1.0)
    parser.add_argument("--w_frame", type=float, default=1.0)
    parser.add_argument("--w_ctr", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_path = args.data_path if args.data_path is not None else resolve_default_data_path()
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")

    print(f"Using data: {data_path}")
    print("Deriving event feature schema...")
    features = derive_event_features(path=data_path, max_rows=args.max_rows)
    print(f"Feature count: {len(features)}")

    print("Building per-feature vocabularies...")
    feature_vocab = build_feature_vocab(
        path=data_path,
        features=features,
        min_freq=args.min_freq,
        max_rows=args.max_rows,
    )
    vocab_sizes = {k: len(v) for k, v in feature_vocab.items()}

    dataset = Events360Dataset(
        path=data_path,
        features=features,
        feature_vocab=feature_vocab,
        max_rows=args.max_rows,
        max_players=args.max_players,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_batch,
    )
    print(f"Loaded rows: {len(dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = EventConditionedSceneEncoder(
        vocab_sizes=vocab_sizes,
        numeric_dim=8,
        player_dim=8,
        d_model=args.d_model,
        out_dim=args.out_dim,
        event_layers=args.event_layers,
        frame_layers=args.frame_layers,
        num_heads=args.num_heads,
    ).to(device)
    mam_head = MaskedFeatureHead(vocab_sizes=vocab_sizes, d_model=args.d_model).to(device)
    frame_head = FrameReconstructionHead(d_model=args.d_model, player_dim=8).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mam_head.parameters()) + list(frame_head.parameters()),
        lr=args.lr,
    )
    feature_list = list(vocab_sizes.keys())
    feature_vocab_sizes = [vocab_sizes[f] for f in feature_list]

    model.train()
    mam_head.train()
    frame_head.train()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        epoch_mam = 0.0
        epoch_frame = 0.0
        epoch_ctr = 0.0

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            feat_ids = batch["feat_ids"].to(device)
            numeric = batch["numeric"].to(device)
            frame_tokens = batch["frame_tokens"].to(device)
            frame_mask = batch["frame_mask"].to(device)

            masked_feat_ids, feature_labels = mask_feature_ids(
                feat_ids=feat_ids,
                feature_vocab_sizes=feature_vocab_sizes,
                mask_prob=args.mask_prob_features,
            )
            masked_frame_positions = mask_frame_positions(
                frame_mask=frame_mask,
                mask_prob=args.mask_prob_players,
            )

            out_1 = model(
                feat_ids=masked_feat_ids,
                numeric_values=numeric,
                frame_tokens=frame_tokens,
                frame_mask=frame_mask,
                frame_mask_positions=masked_frame_positions,
            )

            # View 2 with mild geometric jitter for contrastive learning.
            jittered_frames = jitter_frame_tokens(frame_tokens, frame_mask, noise_std=0.01)
            out_2 = model(
                feat_ids=masked_feat_ids,
                numeric_values=numeric,
                frame_tokens=jittered_frames,
                frame_mask=frame_mask,
                frame_mask_positions=masked_frame_positions,
            )

            logits = mam_head(out_1["feature_tokens"])
            mam_loss = 0.0
            used = 0
            for i, feature in enumerate(feature_list):
                target = feature_labels[:, i]
                if (target != -100).any():
                    mam_loss = mam_loss + F.cross_entropy(
                        logits[feature],
                        target,
                        ignore_index=-100,
                    )
                    used += 1
            if used > 0:
                mam_loss = mam_loss / used
            else:
                mam_loss = torch.tensor(0.0, device=device)

            frame_pred = frame_head(out_1["frame_tokens"])
            if masked_frame_positions.any():
                frame_loss = F.mse_loss(
                    frame_pred[masked_frame_positions],
                    frame_tokens[masked_frame_positions],
                )
            else:
                frame_loss = torch.tensor(0.0, device=device)

            ctr_loss = nt_xent_loss(
                out_1["projection"],
                out_2["projection"],
                temperature=args.temperature,
            )

            total_loss = (
                args.w_mam * mam_loss
                + args.w_frame * frame_loss
                + args.w_ctr * ctr_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += float(total_loss.item())
            epoch_mam += float(mam_loss.item())
            epoch_frame += float(frame_loss.item())
            epoch_ctr += float(ctr_loss.item())

        denom = max(1, len(loader))
        print(
            f"Epoch {epoch + 1}: "
            f"loss={epoch_loss / denom:.4f} "
            f"mam={epoch_mam / denom:.4f} "
            f"frame={epoch_frame / denom:.4f} "
            f"ctr={epoch_ctr / denom:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "mam_head": mam_head.state_dict(),
        "frame_head": frame_head.state_dict(),
        "feature_vocab": feature_vocab,
        "features": features,
        "args": vars(args),
    }
    torch.save(state, args.output)
    print(f"Saved checkpoint to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
