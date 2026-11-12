"""Interrupted baseline training resumes at the next epoch and reproduces an
uninterrupted run exactly."""

import pandas as pd
import torch

from hydrokg.config import load_config
from hydrokg.lstm import train as lstm_train
from tests.synthetic_camels import write_synthetic_camels


def test_resume_reproduces_uninterrupted_training(tmp_path):
    root = tmp_path / "camels"
    basins = write_synthetic_camels(root, n_basins=4)
    cfg = load_config(overrides=["lstm.seq_length=30", "lstm.batch_size=64", "lstm.hidden_size=8"])
    period = (pd.Timestamp("1991-10-01"), pd.Timestamp("1993-09-30"))

    straight = dict(cfg["lstm"], epochs=2)
    lstm_train.train(straight, root, basins, tmp_path / "straight", period, seed=5)

    first = dict(cfg["lstm"], epochs=1)
    lstm_train.train(first, root, basins, tmp_path / "resumed", period, seed=5)
    epoch1_mtime = (tmp_path / "resumed" / "model_epoch1.pt").stat().st_mtime
    lstm_train.train(straight, root, basins, tmp_path / "resumed", period, seed=5)

    assert (tmp_path / "resumed" / "model_epoch1.pt").stat().st_mtime == epoch1_mtime   # epoch 1 not retrained
    a = torch.load(tmp_path / "straight" / "model_epoch2.pt")
    b = torch.load(tmp_path / "resumed" / "model_epoch2.pt")
    for key in a:
        assert torch.equal(a[key], b[key]), key
    assert lstm_train.last_complete_epoch(tmp_path / "resumed", 2) == 2
