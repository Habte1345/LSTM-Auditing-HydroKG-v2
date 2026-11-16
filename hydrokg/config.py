"""Configuration loading: defaults from configs/default.yaml, deep-merged with an
experiment file and optional dotted-key overrides (``key.sub=value``)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _set_dotted(cfg: dict, dotted: str, raw_value: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = yaml.safe_load(raw_value)


def load_config(experiment: Optional[str | Path] = None,
                overrides: Optional[Iterable[str]] = None) -> dict:
    """Return the merged configuration dictionary."""
    with open(DEFAULT_CONFIG) as fp:
        cfg = yaml.safe_load(fp)
    if experiment is not None:
        with open(experiment) as fp:
            cfg = _deep_merge(cfg, yaml.safe_load(fp) or {})
    for item in overrides or []:
        key, _, value = item.partition("=")
        _set_dotted(cfg, key.strip(), value.strip())
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    """Fail early on inconsistent settings."""
    train = period(cfg, "train")
    test = period(cfg, "test")
    if not train[1] < test[0]:
        raise ValueError(f"Training period {train} must end before the test period {test} begins.")
    if cfg["split"]["mode"] not in ("temporal", "basin_holdout"):
        raise ValueError(f"Unknown split.mode {cfg['split']['mode']!r}")
    if cfg["correction"]["method"] not in ("none", "clip", "analog"):
        raise ValueError(f"Unknown correction.method {cfg['correction']['method']!r}")
    from hydrokg.rules.registry import RULES
    for rule_id in cfg["correction"]["daily_rules"]:
        if RULES[rule_id].uses_test_observations:
            raise ValueError(
                f"correction.daily_rules contains {rule_id}, which needs observed discharge; "
                "corrections may only be triggered by observation-free rules.")
    if cfg["burden"]["weighting"] not in ("equal_rules", "equal_classes", "custom"):
        raise ValueError(f"Unknown burden.weighting {cfg['burden']['weighting']!r}")
    if not 0 < cfg["fine_tune"]["epochs"] < cfg["lstm"]["epochs"]:
        raise ValueError("fine_tune.epochs must be between 1 and lstm.epochs - 1: the scenarios "
                         "re-train the final fine_tune.epochs epochs of the baseline.")
    profile_rules = cfg["fine_tune"]["profile_feature"]["rules"]
    if profile_rules not in ("auto", "all", "observation_free"):
        raise ValueError(f"Unknown fine_tune.profile_feature.rules {profile_rules!r}")
    if cfg["split"]["mode"] == "basin_holdout" and profile_rules == "all":
        raise ValueError("profile_feature.rules='all' needs observed discharge in every basin, "
                         "which held-out basins do not have; use 'observation_free' or 'auto'.")


def period(cfg: dict, name: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start, end = cfg["periods"][name]
    return pd.Timestamp(start), pd.Timestamp(end)


def profile_rule_set(cfg: dict) -> str:
    choice = cfg["fine_tune"]["profile_feature"]["rules"]
    if choice == "auto":
        return "observation_free" if cfg["split"]["mode"] == "basin_holdout" else "all"
    return choice


def config_hash(cfg: dict) -> str:
    """Short, stable hash of a configuration, recorded in provenance."""
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_config(cfg: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fp:
        yaml.safe_dump(cfg, fp, sort_keys=False)
