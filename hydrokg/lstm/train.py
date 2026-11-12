"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Herrnegger, M., Sampson, A. K., Hochreiter, S., & Nearing, G. S. ( 2019).
Toward improved predictions in ungauged basins: Exploiting the power of machine learning.
Water Resources Research, 55. https://doi.org/10.1029/2019WR026065

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>

Training and inference routines of HydroAuditToolFrameowrk/src/main.py. The numerical
procedure (data pipeline, model, NSE loss, Adam, gradient-norm clipping, learning-rate
schedule, many-to-one inference, output rescaling) is unchanged. Changes are limited to
how settings are supplied (configuration dictionary instead of module globals and
command-line flags), an explicit basin list and run directory, a configurable evaluation
period and checkpoint, an explicit `device` argument, and saving the optimizer and random
number generator states with every epoch so that an interrupted training resumes at the next
epoch. See hydrokg/lstm/VENDORED.md.
"""
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .datasets import CamelsH5, CamelsTXT
from .datautils import add_camels_attributes, load_attributes, rescale_features
from .model import Model
from .nseloss import NSELoss
from .utils import create_h5_files

N_DYNAMIC_INPUTS = 5


def set_seeds(seed: int) -> None:
    """Fix random seeds exactly as in main.train()."""
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def n_static_attributes(db_path: str, basins: List) -> int:
    """Number of static attributes CamelsH5/CamelsTXT will concatenate (27 for CAMELS)."""
    return load_attributes(db_path, basins, drop_lat_lon=True).shape[1]


def input_size(lstm_cfg: Dict, db_path: str, basins: List) -> int:
    """main.py hard-codes 5 or 32 (= 5 forcings + 27 attributes); computed here so that
    additional static columns (the violation profile) are counted."""
    if lstm_cfg["no_static"] or not lstm_cfg["concat_static"]:
        return N_DYNAMIC_INPUTS
    return N_DYNAMIC_INPUTS + n_static_attributes(db_path, basins)


def build_model(lstm_cfg: Dict, n_inputs: int, device: torch.device) -> Model:
    return Model(
        input_size_dyn=n_inputs,
        hidden_size=lstm_cfg["hidden_size"],
        initial_forget_bias=lstm_cfg["initial_forget_gate_bias"],
        dropout=lstm_cfg["dropout"],
        concat_static=lstm_cfg["concat_static"],
        no_static=lstm_cfg["no_static"]).to(device)


def prepare_run(lstm_cfg: Dict, camels_root: Path, basins: List, run_dir: Path,
                train_period: Tuple[pd.Timestamp, pd.Timestamp], seed: int) -> Dict:
    """Equivalent of main._setup_run + main._prepare_data for an explicit run directory."""
    run_dir = Path(run_dir)
    train_dir = run_dir / "data" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = dict(lstm_cfg)
    run_cfg.update({
        "camels_root": str(camels_root), "run_dir": str(run_dir), "seed": seed,
        "train_start": train_period[0].strftime("%d%m%Y"),
        "train_end": train_period[1].strftime("%d%m%Y"),
        "db_path": str(run_dir / "attributes.db"),
        "train_file": str(train_dir / "train_data.h5"),
        "basins": list(basins),
    })
    if not Path(run_cfg["db_path"]).exists():
        add_camels_attributes(camels_root, db_path=run_cfg["db_path"])
    if not Path(run_cfg["train_file"]).exists():
        create_h5_files(
            camels_root=Path(camels_root),
            out_file=Path(run_cfg["train_file"]),
            basins=basins,
            dates=[train_period[0], train_period[1]],
            with_basin_str=True,
            seq_length=lstm_cfg["seq_length"])
    run_cfg["lr_schedule"] = {str(k): v for k, v in lstm_cfg["lr_schedule"].items()}
    with (run_dir / "cfg.json").open("w") as fp:
        json.dump(run_cfg, fp, sort_keys=True, indent=4)
    return run_cfg


def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_func: nn.Module,
                loader: DataLoader, cfg: Dict, epoch: int, use_mse: bool, device: torch.device):
    """Train model for a single epoch (unchanged from main.train_epoch)."""
    model.train()

    # process bar handle
    pbar = tqdm(loader, file=sys.stdout)
    pbar.set_description(f'# Epoch {epoch}')

    # Iterate in batches over training set
    for data in pbar:
        # delete old gradients
        optimizer.zero_grad()

        # forward pass through LSTM
        x, y, q_stds = data
        x, y, q_stds = x.to(device), y.to(device), q_stds.to(device)
        predictions = model(x)[0]

        # MSELoss
        if use_mse:
            loss = loss_func(predictions, y)

        # NSELoss needs std of each basin for each sample
        else:
            q_stds = q_stds.to(device)
            loss = loss_func(predictions, y, q_stds)

        # calculate gradients
        loss.backward()

        if cfg["clip_norm"]:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_value"])

        # perform parameter update
        optimizer.step()

        pbar.set_postfix_str(f"Loss: {loss.item():5f}")


def train(lstm_cfg: Dict, camels_root: Path, basins: List, run_dir: Path,
          train_period: Tuple[pd.Timestamp, pd.Timestamp], seed: int,
          device: Optional[torch.device] = None) -> Dict:
    """Train the baseline LSTM (main.train with explicit arguments)."""
    device = device or torch.device(lstm_cfg.get("device", "cpu"))
    set_seeds(seed)
    run_cfg = prepare_run(lstm_cfg, camels_root, basins, run_dir, train_period, seed)

    ds = CamelsH5(
        h5_file=Path(run_cfg["train_file"]),
        basins=basins,
        db_path=run_cfg["db_path"],
        concat_static=lstm_cfg["concat_static"],
        cache=lstm_cfg["cache_data"],
        no_static=lstm_cfg["no_static"])
    loader = DataLoader(
        ds, batch_size=lstm_cfg["batch_size"], shuffle=True, num_workers=lstm_cfg["num_workers"])

    model = build_model(lstm_cfg, input_size(lstm_cfg, run_cfg["db_path"], basins), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"])
    loss_func = nn.MSELoss() if lstm_cfg["use_mse"] else NSELoss()

    # reduce learning rates after each 10 epochs
    learning_rates = {int(k): v for k, v in lstm_cfg["lr_schedule"].items()}

    # resume from the last epoch whose model, optimizer and RNG states were all saved
    start_epoch = 1
    last = last_complete_epoch(run_dir, lstm_cfg["epochs"])
    if last is not None:
        model.load_state_dict(torch.load(Path(run_dir) / f"model_epoch{last}.pt", map_location=device))
        optimizer.load_state_dict(torch.load(Path(run_dir) / f"optimizer_epoch{last}.pt", map_location=device))
        restore_rng_state(torch.load(Path(run_dir) / f"rng_epoch{last}.pt", weights_only=False))
        start_epoch = last + 1
        print(f"Resuming training at epoch {start_epoch} from {run_dir}")

    for epoch in range(start_epoch, lstm_cfg["epochs"] + 1):
        # set new learning rate
        if epoch in learning_rates.keys():
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rates[epoch]

        train_epoch(model, optimizer, loss_func, loader, lstm_cfg, epoch, lstm_cfg["use_mse"], device)

        model_path = Path(run_dir) / f"model_epoch{epoch}.pt"
        torch.save(optimizer.state_dict(), str(Path(run_dir) / f"optimizer_epoch{epoch}.pt"))
        torch.save(rng_state(), str(Path(run_dir) / f"rng_epoch{epoch}.pt"))
        torch.save(model.state_dict(), str(model_path))   # written last: marks the epoch as complete
    return run_cfg


def rng_state() -> Dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def last_complete_epoch(run_dir: Path, max_epoch: int) -> Optional[int]:
    """Highest epoch <= max_epoch with model, optimizer and RNG states saved, or None."""
    done = [e for e in range(1, max_epoch + 1)
            if all((Path(run_dir) / f"{name}_epoch{e}.pt").exists() for name in ("model", "optimizer", "rng"))]
    return max(done) if done else None


def evaluate_basin(model: nn.Module, loader: DataLoader,
                   device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate model on a single basin (unchanged from main.evaluate_basin)."""
    model.eval()

    preds, obs = None, None

    with torch.no_grad():
        for data in loader:
            if len(data) == 3:
                x, y, _ = data
            else:
                x, y = data

            x, y = x.to(device), y.to(device)
            p = model(x)[0]

            if preds is None:
                preds = p.detach().cpu()
                obs = y.detach().cpu()
            else:
                preds = torch.cat((preds, p.detach().cpu()), 0)
                obs = torch.cat((obs, y.detach().cpu()), 0)

        preds = rescale_features(preds.numpy(), variable='output')
        obs = obs.numpy()

    return preds, obs


def attribute_scaling(train_file: Path, db_path: str, basins: List,
                      concat_static: bool) -> Tuple[pd.Series, pd.Series]:
    """Attribute means/stds of the training basins, as in main.evaluate()."""
    ds_train = CamelsH5(h5_file=Path(train_file), db_path=db_path, basins=basins,
                        concat_static=concat_static)
    return ds_train.get_attribute_means(), ds_train.get_attribute_stds()


def predict(model: nn.Module, lstm_cfg: Dict, camels_root: Path, basins: List,
            period: Tuple[pd.Timestamp, pd.Timestamp], db_path: str,
            means: pd.Series, stds: pd.Series, device: torch.device,
            show_progress: bool = True) -> Dict[str, pd.DataFrame]:
    """main.evaluate() for an arbitrary period. Returns {basin: DataFrame(qobs, qsim)} with
    qobs in mm/day exactly as read by CamelsTXT (missing values are NOT masked here; see
    hydrokg.data.predictions.mask_missing)."""
    date_range = pd.date_range(start=period[0], end=period[1])
    results = {}
    iterator = tqdm(basins) if show_progress else basins
    for basin in iterator:
        ds_test = CamelsTXT(
            camels_root=Path(camels_root),
            basin=basin,
            dates=[period[0], period[1]],
            is_train=False,
            seq_length=lstm_cfg["seq_length"],
            with_attributes=True,
            attribute_means=means,
            attribute_stds=stds,
            concat_static=lstm_cfg["concat_static"],
            db_path=db_path)
        loader = DataLoader(ds_test, batch_size=1024, shuffle=False, num_workers=0)

        preds, obs = evaluate_basin(model, loader, device)

        # preds and obs are already 1-to-1 daily predictions -> just flatten & align
        preds = preds.flatten()[:len(date_range)]
        obs = obs.flatten()[:len(date_range)]
        results[basin] = pd.DataFrame({'qobs': obs, 'qsim': preds}, index=date_range[:len(preds)])
    return results
