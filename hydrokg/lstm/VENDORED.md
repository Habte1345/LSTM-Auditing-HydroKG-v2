# Vendored baseline LSTM

Source: <https://github.com/Habte1345/HydroAuditToolFrameowrk> (commit `247f150`), which
builds on the code accompanying Kratzert et al. (2019, WRR, doi:10.1029/2019WR026065) and
Kratzert et al. (2018, HESS, doi:10.5194/hess-22-6005-2018). Licensed under Apache-2.0
(see the file headers).

| File here | Source file | SHA-256 of source | Changes |
|---|---|---|---|
| `lstm.py` | `Scripts/lstm.py` | `35abcfb1…` | none (byte-identical) |
| `datasets.py` | `data/datasets.py` | `76afbfc1…` | none (byte-identical) |
| `datautils.py` | `data/datautils.py` | `1d776729…` | removed the unused module-level line `camels_root = r"F:/Data/CAMEL_SI/CAMELS_US/"` |
| `nseloss.py` | `Scripts/nseloss.py` | `03032940…` | kept `NSELoss` unchanged; removed the unused `HydroKGRealTimeLoss` class (a rule-penalty loss that this study does not use) |
| `utils.py` | `Scripts/utils.py` | `1689605d…` | relative import of `CamelsTXT`; `get_basin_list` takes the basin file as an argument and skips blank lines |
| `model.py` | `src/main.py` (`Model`) | `5ee13611…` | none to the class |
| `train.py` | `src/main.py` (`train`, `train_epoch`, `evaluate`, `evaluate_basin`) | `5ee13611…` | settings passed as a dictionary; explicit basin list, run directory, seed and device; evaluation period and checkpoint are arguments instead of the fixed validation period and `model_epoch5.pt`; the input size is computed from the attribute table instead of the literal 32 (identical for the baseline: 5 forcings + 27 attributes); the prediction index is cut to the number of predictions |

Numerical behaviour is unchanged: identical data pipeline (NLDAS-extended forcing, global
SCALER normalization, 27 static attributes concatenated at every time step), model,
basin-normalized NSE loss (eps = 0.1), Adam, gradient-norm clipping at 1, learning-rate
schedule {11: 5e-4, 21: 1e-4}, many-to-one inference and output rescaling.

Known property of the original evaluation, handled outside this package: `CamelsTXT` with
`is_train=False` does not remove CAMELS missing-discharge codes (-999 cfs), so `qobs` in
prediction files contains large negative values on missing days. `hydrokg.data.predictions.mask_missing`
sets these to NaN before any metric or rule is computed.
