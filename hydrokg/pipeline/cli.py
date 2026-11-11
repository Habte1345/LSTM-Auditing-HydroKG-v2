"""Command-line interface.

    hydrokg run        --experiment configs/experiments/temporal.yaml --seed 658666 [--set key=value ...]
    hydrokg aggregate  --experiment-dir results/hydrokg_v2/temporal
    hydrokg reaudit    --predictions old.p [--predictions other.p] --run-dir <baseline run> --out <dir>
    hydrokg leakage    --kg results/.../kg/hydrokg.ttl [--holdout]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path


def _cmd_run(args):
    from hydrokg.config import load_config
    from hydrokg.pipeline.experiment import Experiment
    cfg = load_config(args.experiment, args.set)
    seeds = [args.seed] if args.seed is not None else cfg["seeds"]
    for seed in seeds:
        Experiment(cfg, seed).run()


def _cmd_aggregate(args):
    from hydrokg.pipeline.analysis import aggregate_seeds
    print(aggregate_seeds(args.experiment_dir))


def _cmd_reaudit(args):
    from hydrokg.config import load_config
    from hydrokg.pipeline.reaudit import reaudit
    cfg = load_config(args.experiment, args.set)
    reaudit(cfg, args.predictions, args.labels, args.run_dir, args.out)


def _cmd_leakage(args):
    from hydrokg.kg import store
    from hydrokg.kg.provenance import leakage_report
    report = leakage_report(store.load(args.kg), args.holdout)
    print("No leakage found." if report.empty else report.to_string(index=False))


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="hydrokg")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a full experiment")
    p.add_argument("--experiment", required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. paths.camels_root=/data/CAMELS_US")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("aggregate", help="aggregate analysis tables over seeds")
    p.add_argument("--experiment-dir", required=True)
    p.set_defaults(func=_cmd_aggregate)

    p = sub.add_parser("reaudit", help="audit existing prediction pickles with the revised rules")
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--run-dir", required=True, help="baseline run directory (attributes.db)")
    p.add_argument("--out", required=True)
    p.add_argument("--experiment", default=None)
    p.add_argument("--set", nargs="*", default=[])
    p.set_defaults(func=_cmd_reaudit)

    p = sub.add_parser("leakage", help="run the provenance leakage query on a saved KG")
    p.add_argument("--kg", required=True)
    p.add_argument("--holdout", action="store_true")
    p.set_defaults(func=_cmd_leakage)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
