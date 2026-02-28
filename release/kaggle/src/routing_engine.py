from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from structural_profile import (
    ProfileThresholds,
    extract_source_features,
    label_features,
    route_decision,
)


ROUTE_TO_POLICY = {
    "RETRIEVE_INTERNAL": "internal_only",
    "RETRIEVE_HYBRID": "hybrid",
    "RETRIEVE_ORACC_FALLBACK": "fallback",
    "RERANK_STRONG": "strong_rerank",
}


@dataclass(frozen=True)
class LoadedRoutingMap:
    path: Path
    sha256: str
    payload: dict[str, Any]
    routing_map: dict[str, Any]
    profile_thresholds: ProfileThresholds


def _load_json(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Routing map must be a JSON object: {path}")
    return obj


def _parse_thresholds(raw: dict[str, Any]) -> ProfileThresholds:
    return ProfileThresholds(
        fragment_bracket_p90=float(raw["fragment_bracket_p90"]),
        fragment_unknown_p90=float(raw["fragment_unknown_p90"]),
        formula_high_p85=float(raw["formula_high_p85"]),
        formula_mid_p60=float(raw["formula_mid_p60"]),
        numeric_high_p85=float(raw["numeric_high_p85"]),
        pn_high_p85=float(raw["pn_high_p85"]),
    )


def resolve_routing_map_path(path: Path | None, repo_root: Path) -> Path:
    if path is not None:
        return path if path.is_absolute() else (repo_root / path)
    candidates = [
        repo_root / "routing_map.json",
        repo_root / "artifacts" / "routing_map.json",
        repo_root / "artifacts" / "profiles" / "routing_map.json",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "Routing map not found. Provide --routing-map or create one with scripts/profile_corpus.py."
    )


def load_routing_map(path: Path) -> LoadedRoutingMap:
    rpath = path.resolve()
    payload = _load_json(rpath)
    if "routing_map" in payload:
        routing_map = payload["routing_map"]
        thresholds_raw = payload.get("thresholds", {})
    else:
        routing_map = payload
        thresholds_raw = payload.get("thresholds", {})
    if not isinstance(routing_map, dict):
        raise ValueError(f"Invalid routing map payload in: {rpath}")
    if "thresholds" not in routing_map or "routes" not in routing_map:
        raise ValueError(f"Routing map missing required keys (thresholds/routes): {rpath}")
    if not thresholds_raw:
        raise ValueError(f"Profile thresholds missing in routing map payload: {rpath}")
    profile_thresholds = _parse_thresholds(thresholds_raw)
    return LoadedRoutingMap(
        path=rpath,
        sha256=sha256(rpath.read_bytes()).hexdigest(),
        payload=payload,
        routing_map=routing_map,
        profile_thresholds=profile_thresholds,
    )


def profile_source(
    text: str,
    *,
    token_freq: dict[str, int],
    profile_thresholds: ProfileThresholds,
) -> tuple[dict[str, float], dict[str, Any]]:
    features = extract_source_features(str(text or ""), token_freq=token_freq)
    labels = label_features(features, profile_thresholds)
    return features, labels


def choose_policy(
    *,
    labels: dict[str, Any],
    internal_top_score: float | None,
    internal_gap: float | None,
    routing_map: dict[str, Any],
) -> dict[str, Any]:
    decision = route_decision(
        labels=labels,
        internal_top_score=internal_top_score,
        internal_gap=internal_gap,
        routing_map=routing_map,
    )
    route = str(decision["route"])
    if route not in ROUTE_TO_POLICY:
        raise ValueError(f"Unknown route from routing map: {route}")
    return {
        "route": route,
        "policy_name": ROUTE_TO_POLICY[route],
        "policy_params": dict(decision.get("policy_params", {})),
        "rationale": str(decision.get("rationale", "")),
    }

