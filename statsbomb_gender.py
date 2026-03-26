from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen


RAW_BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _load_json_from_path(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_json_from_url(url: str):
    with urlopen(url, timeout=30) as response:
        return json.load(response)


def _iter_json_objects(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        while True:
            chunk = f.read(1 << 20)
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
                    break
                yield obj
                buf = buf[idx:]
        buf = buf.lstrip()
        if buf:
            obj, _ = decoder.raw_decode(buf)
            yield obj


def _sorted_player_ids(player_ids: Iterable[str]) -> list[str]:
    def sort_key(value: str):
        try:
            return (0, int(value))
        except (TypeError, ValueError):
            return (1, str(value))

    return sorted((str(pid) for pid in player_ids), key=sort_key)


def majority_label(counter: Counter | None) -> str | None:
    if not counter:
        return None
    return counter.most_common(1)[0][0]


def build_match_gender_map(local_root: str | Path | None = None, cache_path: str | Path | None = None) -> dict[str, str]:
    cache_file = Path(cache_path) if cache_path else None
    if cache_file and cache_file.exists():
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    local_root = Path(local_root) if local_root else None
    competitions_path = local_root / "competitions.json" if local_root else None
    if competitions_path and competitions_path.exists():
        competitions = _load_json_from_path(competitions_path)
        use_local = True
    else:
        competitions = _load_json_from_url(f"{RAW_BASE_URL}/competitions.json")
        use_local = False

    match_gender_map: dict[str, str] = {}
    for comp in competitions:
        competition_id = comp["competition_id"]
        season_id = comp["season_id"]
        gender = comp.get("competition_gender")
        if gender is None:
            continue

        if use_local:
            matches_path = local_root / "matches" / str(competition_id) / f"{season_id}.json"
            if not matches_path.exists():
                continue
            matches = _load_json_from_path(matches_path)
        else:
            matches = _load_json_from_url(f"{RAW_BASE_URL}/matches/{competition_id}/{season_id}.json")

        for match in matches:
            match_id = match.get("match_id")
            if match_id is not None:
                match_gender_map[str(match_id)] = gender

    if cache_file:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(match_gender_map, f, ensure_ascii=False, indent=2)

    return match_gender_map


def build_player_metadata(
    data_path: str | Path,
    player_ids: Iterable[str],
    match_gender_map: dict[str, str],
    max_rows: int | None = None,
) -> dict[str, dict[str, str | None]]:
    target_ids = set(str(pid) for pid in player_ids)
    player_names: dict[str, str] = {}
    gender_counts: dict[str, Counter] = defaultdict(Counter)
    rows_seen = 0

    for ev in _iter_json_objects(Path(data_path)):
        player_id = ev.get("player.id")
        match_id = ev.get("match_id")
        if player_id is None or match_id is None:
            continue
        player_id = str(player_id)
        if target_ids and player_id not in target_ids:
            continue

        player_name = ev.get("player.name")
        if player_name:
            player_names[player_id] = player_name

        gender = match_gender_map.get(str(match_id))
        if gender:
            gender_counts[player_id][gender] += 1

        rows_seen += 1
        if max_rows is not None and rows_seen >= max_rows:
            break

    metadata: dict[str, dict[str, str | None]] = {}
    for player_id in _sorted_player_ids(target_ids):
        metadata[player_id] = {
            "name": player_names.get(player_id),
            "gender": majority_label(gender_counts.get(player_id)),
        }
    return metadata
