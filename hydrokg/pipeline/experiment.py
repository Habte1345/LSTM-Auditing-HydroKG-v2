"""End-to-end experiment for one seed:

  split -> baseline LSTM -> basin contexts (training period) -> KG (catchments, inferred
  applicability) -> baseline predictions (train, test) -> audits -> fine-tuning arms ->
  test predictions -> corrections -> audits -> provenance + leakage check -> analysis tables

Every step writes its output before the next starts and is skipped if the output exists,
so an interrupted job resumes where it stopped.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdflib import URIRef

from hydrokg.config import config_hash, period, profile_rule_set, save_config
from hydrokg.data import camels
from hydrokg.data.predictions import attach_series, load_predictions, mask_missing, missing_report, save_predictions
from hydrokg.data.splits import describe_split, save_split, stratified_holdout, temporal_split
from hydrokg.enhance.correction import apply_correction
from hydrokg.enhance.feedback import require_training_period
from hydrokg.enhance.finetune import counts_to_results, fine_tune
from hydrokg.kg import store
from hydrokg.kg.analogs import AnalogFinder, benchmark
from hydrokg.kg.graph import HK, apply_kg_applicability, build_catchment_graph
from hydrokg.kg.provenance import Provenance, assert_no_leakage, leakage_report
from hydrokg.kg.rules_rdf import shapes_turtle
from hydrokg.lstm import train as lstm_train
from hydrokg.lstm.utils import get_basin_list
from hydrokg.pipeline.audit import audit_predictions
from hydrokg.rules.context import build_context, regional_omega

log = logging.getLogger("hydrokg")

ARM_FLAGS = {"control": (False, False), "curriculum": (True, False),
             "profile": (False, True), "curriculum_profile": (True, True)}


def _pickle(path: Path, obj=None):
    if obj is None:
        with open(path, "rb") as fp:
            return pickle.load(fp)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)
    return obj


class Experiment:
    def __init__(self, cfg: dict, seed: int):
        self.cfg = cfg
        self.seed = seed
        self.camels_root = Path(cfg["paths"]["camels_root"])
        self.out = Path(cfg["paths"]["output_dir"]) / cfg["experiment"]["name"] / cfg["split"]["mode"] / f"seed_{seed}"
        self.out.mkdir(parents=True, exist_ok=True)
        save_config(cfg, self.out / "config.yaml")
        self.train_period = period(cfg, "train")
        self.test_period = period(cfg, "test")
        self.holdout = cfg["split"]["mode"] == "basin_holdout"
        self.device = torch.device(cfg["lstm"].get("device", "cpu"))

    # ------------------------------------------------------------------ setup
    def basins_and_split(self) -> dict:
        path = self.out / "split.p"
        if path.exists():
            self.split = _pickle(path)
        else:
            basins = get_basin_list(self.cfg["paths"]["basin_list"])
            if self.holdout:
                attrs = self._attributes_from_txt(basins)
                strata = attrs["aridity"].apply(lambda a: _aridity_class(a, self.cfg))
                self.split = stratified_holdout(basins, strata, self.cfg["split"]["holdout_fraction"],
                                                self.cfg["split"]["seed"])
                describe_split(self.split, strata).to_csv(self.out / "split_summary.csv", index=False)
            else:
                self.split = temporal_split(basins)
            _pickle(path, self.split)
        self.train_basins = self.split[0]["train"]
        self.test_basins = self.split[0]["test"]
        self.all_basins = sorted(set(self.train_basins) | set(self.test_basins))
        return self.split

    def _attributes_from_txt(self, basins) -> pd.DataFrame:
        frames = []
        for f in (self.camels_root / "camels_attributes_v2.0").glob("camels_*.txt"):
            frames.append(pd.read_csv(f, sep=";", dtype={"gauge_id": str}).set_index("gauge_id"))
        return pd.concat(frames, axis=1).reindex(basins)

    def baseline(self) -> dict:
        reuse = self.cfg["experiment"]["reuse_baseline_run_dir"]
        run_dir = Path(reuse) if reuse else self.out / "baseline"
        epoch = self.cfg["experiment"]["baseline_checkpoint_epoch"] or self.cfg["lstm"]["epochs"]
        ckpt = run_dir / f"model_epoch{epoch}.pt"
        if not ckpt.exists():
            lstm_train.train(self.cfg["lstm"], self.camels_root, self.train_basins, run_dir,
                             self.train_period, self.seed, self.device)
        with open(run_dir / "cfg.json") as fp:
            self.run_cfg = json.load(fp)
        self.run_cfg["db_path"] = str(run_dir / "attributes.db")
        self.run_cfg["train_file"] = str(run_dir / "data" / "train" / "train_data.h5")
        if not Path(self.run_cfg["train_file"]).exists():
            from hydrokg.lstm.utils import create_h5_files
            Path(self.run_cfg["train_file"]).parent.mkdir(parents=True, exist_ok=True)
            create_h5_files(camels_root=self.camels_root, out_file=Path(self.run_cfg["train_file"]),
                            basins=self.train_basins, dates=[self.train_period[0], self.train_period[1]],
                            with_basin_str=True, seq_length=self.cfg["lstm"]["seq_length"])
        self.checkpoint = ckpt
        return self.run_cfg

    # ------------------------------------------------------------------ contexts and KG
    def contexts(self):
        path = self.out / "contexts.p"
        if path.exists():
            self.train_frames, self.ctx_eval, self.ctx_feedback = _pickle(path)
        else:
            attrs = camels.static_attributes(self.run_cfg["db_path"], self.all_basins)
            self.train_frames = {b: camels.basin_period_frame(self.camels_root, b, self.train_period,
                                                              self.cfg["qc"]["qobs_missing_below"])
                                 for b in self.all_basins}
            # evaluation contexts: thresholds for SCORING (training-period obs of every basin)
            self.ctx_eval = {b: build_context(b, attrs.loc[b], self.train_frames[b], self.cfg, True)
                             for b in self.all_basins}
            # feedback contexts: no observation of held-out basins
            obs_basins = set(self.train_basins)
            fb = {b: build_context(b, attrs.loc[b], self.train_frames[b], self.cfg, b in obs_basins)
                  for b in self.all_basins}
            fallback = regional_omega({b: c for b, c in fb.items() if b in obs_basins})
            for b in self.all_basins:
                if b not in obs_basins:
                    fb[b] = build_context(b, attrs.loc[b], self.train_frames[b], self.cfg, False,
                                          omega_fallback=fallback.get(fb[b].aridity_class))
            self.ctx_feedback = fb
            _pickle(path, (self.train_frames, self.ctx_eval, self.ctx_feedback))
        pd.DataFrame([c.to_dict() for c in self.ctx_eval.values()]).to_csv(self.out / "contexts_eval.csv", index=False)

        level_attrs = camels.model_attributes(self.run_cfg["db_path"], self.all_basins)
        self.kg = build_catchment_graph(self.ctx_feedback.values(), self.cfg, level_attrs)
        self.kg_eval = (self.kg if not self.holdout
                        else build_catchment_graph(self.ctx_eval.values(), self.cfg, level_attrs))
        apply_kg_applicability(self.ctx_feedback, self.kg)
        apply_kg_applicability(self.ctx_eval, self.kg_eval)
        (self.out / "kg").mkdir(exist_ok=True)
        (self.out / "kg" / "shapes.ttl").write_text(shapes_turtle(self.cfg))
        self.finder = AnalogFinder(self.kg, level_attrs, self.cfg["analogs"]["ppr_alpha"])
        self.prov = Provenance(self.kg, f"https://w3id.org/hydrokg/run/{self.cfg['experiment']['name']}/{self.seed}")
        self._provenance_roots()

    def _provenance_roots(self):
        p, tr, te = self.prov, self.train_period, self.test_period
        bset_train = "train" if self.holdout else "all"
        self.e_obs_train = p.dataset(HK.DischargeObservations, "discharge_train", tr, bset_train)
        self.e_obs_test = p.dataset(HK.DischargeObservations, "discharge_test", te, "all")
        self.e_forcing = p.dataset(HK.ForcingData, "forcing_nldas_extended", (tr[0], te[1]))
        self.e_attrs = p.dataset(HK.StaticAttributes, "camels_attributes_v2")
        a_train = p.activity(HK.TrainingActivity, "baseline_training",
                             [self.e_obs_train, self.e_forcing, self.e_attrs],
                             seed=self.seed, configHash=config_hash(self.cfg))
        self.e_ckpt = p.entity(HK.ModelCheckpoint, "baseline_checkpoint", a_train, basin_set=bset_train)

    # ------------------------------------------------------------------ predictions and audits
    def predict(self, checkpoint: Path, db_path: str, basins: list[str], which: str, label: str) -> dict:
        path = self.out / "predictions" / f"{label}_{which}.p"
        if path.exists():
            return _pickle(path)
        lstm_cfg = self.cfg["lstm"]
        model = lstm_train.build_model(lstm_cfg, lstm_train.input_size(lstm_cfg, db_path, self.train_basins), self.device)
        model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        means, stds = lstm_train.attribute_scaling(self.run_cfg["train_file"], db_path, self.train_basins,
                                                   lstm_cfg["concat_static"])
        per = self.train_period if which == "train" else self.test_period
        preds = lstm_train.predict(model, lstm_cfg, self.camels_root, basins, per, db_path, means, stds, self.device)
        path.parent.mkdir(parents=True, exist_ok=True)
        missing_report(preds, self.cfg["qc"]["qobs_missing_below"]).to_csv(
            self.out / "predictions" / f"{label}_{which}_missing.csv", index=False)
        preds = mask_missing(preds, self.cfg["qc"]["qobs_missing_below"])
        preds = attach_series(preds, {b: camels.precipitation(self.camels_root, b) for b in basins}, "p")
        return _pickle(path, preds)

    def audit(self, preds: dict, label: str, feedback: bool = False) -> dict:
        path = self.out / "audits" / f"{label}.p"
        if path.exists():
            return _pickle(path)
        ctx = self.ctx_feedback if feedback else self.ctx_eval
        graph = self.kg if feedback else self.kg_eval
        out = audit_predictions(preds, ctx, self.cfg, graph, workers=self.cfg["experiment"]["audit_workers"],
                                keep_flags=label.startswith("baseline_test"), desc=label)
        path.parent.mkdir(parents=True, exist_ok=True)
        out["table"].to_csv(self.out / "audits" / f"{label}.csv")
        if len(out["validity"]):
            out["validity"].to_csv(self.out / "audits" / f"{label}_observation_validity.csv")
        return _pickle(path, out)

    def _record_audit(self, label: str, result: dict, used: list, period_: tuple, basin_set: str) -> URIRef:
        act = self.prov.activity(HK.AuditActivity, f"audit_{label}", used)
        store.add_audit_outcomes(self.kg, act, result["results"])
        store.add_violation_facts(self.kg, act, result["records"], self.cfg["kg"]["store_daily_violations"])
        return self.prov.entity(HK.AuditReport, f"report_{label}", act, period_, basin_set)

    # ------------------------------------------------------------------ main
    def run(self) -> None:
        cfg = self.cfg
        self.basins_and_split()
        self.baseline()
        self.contexts()
        tr, te = self.train_period, self.test_period
        test_set = "test" if self.holdout else "all"

        # baseline predictions and audits
        pred_train = self.predict(self.checkpoint, self.run_cfg["db_path"], self.all_basins, "train", "baseline")
        a_pt = self.prov.activity(HK.PredictionActivity, "predict_baseline_train", [self.e_ckpt, self.e_forcing, self.e_attrs])
        e_pt = self.prov.entity(HK.PredictionSet, "pred_baseline_train", a_pt, tr)
        # training-period audit used for feedback: observations of training basins only
        fb_preds = {b: (df if b in self.train_basins else df.assign(qobs=np.nan)) for b, df in pred_train.items()}
        audit_train = self.audit(fb_preds, "baseline_train_feedback", feedback=True)
        require_training_period(tr, tr, "feedback audit")
        e_audit_train = self._record_audit("baseline_train_feedback", audit_train, [e_pt, self.e_obs_train, self.e_forcing], tr, "all")
        a_fb = self.prov.activity(HK.FeedbackActivity, "feedback_from_training_audit", [e_audit_train])
        e_fb = self.prov.entity(HK.FeedbackSignal, "feedback_signals", a_fb, tr)
        initial_rates = pd.DataFrame({r.basin_id: r.rates() for r in audit_train["results"]}).T

        pred_test = self.predict(self.checkpoint, self.run_cfg["db_path"], self.test_basins, "test", "baseline")
        a_p = self.prov.activity(HK.PredictionActivity, "predict_baseline_test", [self.e_ckpt, self.e_forcing, self.e_attrs])
        e_p = self.prov.entity(HK.EvaluatedPredictionSet, "pred_baseline_test", a_p, te, test_set)
        self.audit(pred_test, "baseline_test")
        self._record_audit("baseline_test", _pickle(self.out / "audits" / "baseline_test.p"), [e_p, self.e_obs_test], te, test_set)
        for method in cfg["experiment"]["corrections"]:
            if method != "none":
                self._corrected(pred_test, "baseline", method, e_p, test_set)

        floors = {b: c.q_floor for b, c in self.ctx_feedback.items() if b in self.train_basins}
        for arm in cfg["experiment"]["arms"]:
            curriculum, profile = ARM_FLAGS[arm]
            arm_dir = self.out / "arms" / arm
            meta_path = arm_dir / "finetune.json"
            a_ft = self.prov.activity(HK.FineTuningActivity, f"finetune_{arm}",
                                      [self.e_ckpt, e_fb, self.e_obs_train, self.e_forcing, self.e_attrs],
                                      seed=self.seed)

            def record_online(epoch, counts, _arm=arm, _act=a_ft):
                act = self.prov.activity(HK.AuditActivity, f"online_audit_{_arm}_epoch{epoch}", [self.e_obs_train])
                store.add_audit_outcomes(self.kg, act, counts_to_results(counts))

            if meta_path.exists():
                with open(meta_path) as fp:
                    meta = json.load(fp)
            else:
                meta = fine_tune(cfg, self.run_cfg, self.checkpoint, self.train_basins, floors, initial_rates,
                                 curriculum, profile, profile_rule_set(cfg), self.seed, arm_dir,
                                 profile_basins=self.all_basins, on_epoch_end=record_online)
            e_ft = self.prov.entity(HK.ModelCheckpoint, f"checkpoint_{arm}", a_ft)
            preds = self.predict(Path(meta["checkpoint"]), meta["db_path"], self.test_basins, "test", arm)
            a_pa = self.prov.activity(HK.PredictionActivity, f"predict_{arm}_test", [e_ft, e_fb, self.e_forcing, self.e_attrs])
            e_pa = self.prov.entity(HK.EvaluatedPredictionSet, f"pred_{arm}_test", a_pa, te, test_set)
            res = self.audit(preds, f"{arm}_test")
            self._record_audit(f"{arm}_test", res, [e_pa, self.e_obs_test], te, test_set)
            for method in cfg["experiment"]["corrections"]:
                if method != "none":
                    self._corrected(preds, arm, method, e_pa, test_set)

        report = leakage_report(self.kg, self.holdout)
        report.to_csv(self.out / "kg" / "leakage_report.csv", index=False)
        store.save(self.kg, self.out / "kg" / "hydrokg.ttl")
        assert_no_leakage(self.kg, self.holdout)

        if not (self.out / "analysis").exists():
            from hydrokg.pipeline.analysis import analyze_seed
            analyze_seed(self.out, cfg)
        donors = self.train_basins
        flows = {b: f["qobs"] for b, f in self.train_frames.items() if b in donors}
        benchmark(self.finder, flows, donors, donors, cfg["analogs"]["k"]).to_csv(
            self.out / "analysis" / "analog_benchmark.csv", index=False)

    def _corrected(self, preds: dict, label: str, method: str, e_source: URIRef, test_set: str) -> None:
        name = f"{label}+{method}_test"
        cfg = dict(self.cfg)
        cfg["correction"] = dict(cfg["correction"], method=method)
        path = self.out / "predictions" / f"{name}.p"
        if path.exists():
            corrected = _pickle(path)
        else:
            corrected, clog = apply_correction(preds, cfg, self.finder)
            clog.to_csv(self.out / "predictions" / f"{name}_log.csv", index=False)
            _pickle(path, corrected)
        a_c = self.prov.activity(HK.CorrectionActivity, f"correct_{name}", [e_source, self.e_forcing])
        e_c = self.prov.entity(HK.EvaluatedPredictionSet, f"pred_{name}", a_c, self.test_period, test_set)
        res = self.audit(corrected, name)
        self._record_audit(name, res, [e_c, self.e_obs_test], self.test_period, test_set)


def _aridity_class(a, cfg):
    from hydrokg.rules.context import classify_aridity
    s = cfg["stratification"]
    return classify_aridity(float(a), s["aridity_bins"], s["aridity_labels"])
