"""Graph-guided fine-tuning of the baseline LSTM.

Arms (configs/experiments): control (continuation: same optimizer step, uniform
shuffling, unchanged inputs), curriculum, profile, curriculum+profile. Every arm uses the
identical optimization step of the vendored `train_epoch` (NSE loss, Adam, gradient-norm
clipping), the same number of epochs and the same constant learning rate, so differences
between arms are attributable to the feedback mechanisms alone.

Online audit: after each optimizer step, the batch's predictions and targets are rescaled
to mm/day and R0-R3 are evaluated with the basin floors from the training period. The
detections never enter the loss or the backward pass; they set the next epoch's sampling
weights and (for the profile arm) refresh the daily-rule columns of the profile.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from hydrokg.enhance.feedback import (PROFILE_PREFIX, curriculum_weights, profile_matrix,
                                      profile_rules, write_profile_db)
from hydrokg.lstm.datasets import CamelsH5
from hydrokg.lstm.datautils import rescale_features
from hydrokg.lstm.nseloss import NSELoss
from hydrokg.lstm.train import build_model, input_size, set_seeds
from hydrokg.rules.engine import AuditResult, online_batch_counts
from hydrokg.rules.registry import ONLINE_RULES


class IndexedDataset(Dataset):
    """Returns (x, y, q_std, index) so the online audit knows each sample's basin."""

    def __init__(self, ds: CamelsH5):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x, y, q_std = self.ds[idx]
        return x, y, q_std, idx


def warm_start(model: torch.nn.Module, checkpoint: dict) -> dict:
    """Copy every pretrained tensor; for lstm.weight_ih (shape [n_inputs, 4H]) copy the
    pretrained rows and set rows of new inputs to zero, so the model is functionally
    identical to the checkpoint at the start of fine-tuning."""
    state = model.state_dict()
    report = {"copied": [], "extended": []}
    for key, value in checkpoint.items():
        if key not in state:
            continue
        if state[key].shape == value.shape:
            state[key] = value.clone()
            report["copied"].append(key)
        elif key == "lstm.weight_ih" and state[key].shape[1] == value.shape[1] and state[key].shape[0] > value.shape[0]:
            new = torch.zeros_like(state[key])
            new[: value.shape[0]] = value
            state[key] = new
            report["extended"].append(key)
        else:
            raise ValueError(f"Cannot warm-start {key}: {tuple(value.shape)} -> {tuple(state[key].shape)}")
    model.load_state_dict(state)
    return report


def rates_from_counts(counts: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=counts.index)
    for r in ONLINE_RULES:
        ev = counts[f"eval_{r}"].replace(0, np.nan)
        out[r] = counts[f"n_{r}"] / ev
    return out


def counts_to_results(counts: pd.DataFrame) -> list[AuditResult]:
    results = []
    for basin, row in counts.iterrows():
        res = AuditResult(basin_id=str(basin))
        for r in ONLINE_RULES:
            res.counts[r] = int(row[f"n_{r}"])
            res.evaluable[r] = int(row[f"eval_{r}"])
            res.applicable[r] = res.evaluable[r] > 0
        results.append(res)
    return results


def fine_tune(cfg: dict, baseline_run_cfg: dict, checkpoint_path: str | Path, basins: list[str],
              floors: dict[str, float], initial_rates: pd.DataFrame, curriculum: bool, profile: bool,
              profile_mode: str, seed: int, out_dir: str | Path, profile_basins: Optional[list[str]] = None,
              on_epoch_end=None) -> dict:
    """Fine-tune and return paths/metadata.

    initial_rates : basin x rule rates of the TRAINING-period audit of the baseline, as read
        from the knowledge graph (hydrokg.kg.store.feedback_rates); all basins that need a
        profile row, including held-out basins for which only observation-free rules are filled.
    on_epoch_end : callback(epoch, online_counts) -> rates. The experiment pipeline writes the
        epoch's online detections to the knowledge graph as a provenance-tracked audit and
        returns the rates read back from the graph; these rates drive the next epoch. Without
        a callback (standalone use) the rates are computed from the counts directly.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lstm_cfg = dict(cfg["lstm"])
    ft = cfg["fine_tune"]
    device = torch.device(lstm_cfg.get("device", "cpu"))
    set_seeds(seed)

    db_path = baseline_run_cfg["db_path"]
    prof_rules = profile_rules(profile_mode)
    profile_basins = profile_basins or basins
    prof = None
    if profile:
        prof = profile_matrix(initial_rates, profile_basins, prof_rules)
        db_path = str(write_profile_db(baseline_run_cfg["db_path"], out_dir / "attributes_profile.db", prof))

    ds = CamelsH5(h5_file=Path(baseline_run_cfg["train_file"]), basins=basins, db_path=db_path,
                  concat_static=lstm_cfg["concat_static"], cache=lstm_cfg["cache_data"],
                  no_static=lstm_cfg["no_static"])
    model = build_model(lstm_cfg, input_size(lstm_cfg, db_path, basins), device)
    report = warm_start(model, torch.load(checkpoint_path, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=ft["learning_rate"])
    loss_func = NSELoss()
    sample_basins = np.asarray(ds.sample_2_basin)
    online_rates = initial_rates.reindex(columns=ONLINE_RULES)
    history = []

    for epoch in range(1, ft["epochs"] + 1):
        if curriculum:
            w = curriculum_weights(online_rates, basins, ft["curriculum"]["temperature"], ft["curriculum"]["floor"])
            sample_w = torch.tensor(w.reindex(sample_basins).to_numpy(dtype=float, copy=True), dtype=torch.double)
            gen = torch.Generator().manual_seed(seed * 1000 + epoch)
            sampler = WeightedRandomSampler(sample_w, num_samples=len(ds), replacement=True, generator=gen)
            loader = DataLoader(IndexedDataset(ds), batch_size=lstm_cfg["batch_size"], sampler=sampler,
                                num_workers=lstm_cfg["num_workers"])
            w.rename("weight").to_csv(out_dir / f"curriculum_weights_epoch{epoch}.csv", index_label="basin_id")
        else:
            loader = DataLoader(IndexedDataset(ds), batch_size=lstm_cfg["batch_size"], shuffle=True,
                                num_workers=lstm_cfg["num_workers"])

        model.train()
        epoch_counts = []
        for x, y, q_stds, idx in tqdm(loader, desc=f"fine-tune epoch {epoch}", leave=False):
            # --- identical to hydrokg.lstm.train.train_epoch ---
            optimizer.zero_grad()
            x, y, q_stds = x.to(device), y.to(device), q_stds.to(device)
            predictions = model(x)[0]
            loss = loss_func(predictions, y, q_stds)
            loss.backward()
            if lstm_cfg["clip_norm"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), lstm_cfg["clip_value"])
            optimizer.step()
            # --- online audit (detached side channel) ---
            with torch.no_grad():
                q_sim = rescale_features(predictions.detach().cpu().numpy(), "output").ravel()
                q_obs = rescale_features(y.detach().cpu().numpy(), "output").ravel()
            epoch_counts.append(online_batch_counts(sample_basins[idx.numpy()], q_obs, q_sim, floors, cfg["rules"]))

        counts = pd.concat(epoch_counts).groupby(level=0).sum()
        counts.to_csv(out_dir / f"online_counts_epoch{epoch}.csv", index_label="basin_id")
        kg_rates = on_epoch_end(epoch, counts) if on_epoch_end is not None else None
        online_rates = (kg_rates.reindex(columns=ONLINE_RULES) if kg_rates is not None
                        else rates_from_counts(counts))
        online_rates.to_csv(out_dir / f"online_rates_epoch{epoch}.csv", index_label="basin_id")
        history.append({"epoch": epoch, "mean_online_rate": float(online_rates.mean(axis=1).mean()),
                        "feedback_source": "knowledge_graph" if kg_rates is not None else "counts"})

        if profile and epoch < ft["epochs"]:
            refreshed = initial_rates.reindex(profile_basins).copy()
            for r in ONLINE_RULES:
                if r in prof_rules:
                    refreshed.loc[online_rates.index.intersection(refreshed.index), r] = online_rates[r]
            prof = profile_matrix(refreshed, profile_basins, prof_rules)
            write_profile_db(baseline_run_cfg["db_path"], out_dir / "attributes_profile.db", prof)
            ds.db_path = db_path
            ds._load_attributes()

    torch.save(model.state_dict(), out_dir / "model_finetuned.pt")
    if prof is not None:
        prof.to_csv(out_dir / "violation_profile.csv", index_label="basin_id")
    meta = {"checkpoint": str(out_dir / "model_finetuned.pt"), "db_path": db_path, "warm_start": report,
            "curriculum": curriculum, "profile": profile, "profile_rules": prof_rules if profile else [],
            "history": history, "seed": seed}
    with open(out_dir / "finetune.json", "w") as fp:
        json.dump(meta, fp, indent=2)
    return meta
