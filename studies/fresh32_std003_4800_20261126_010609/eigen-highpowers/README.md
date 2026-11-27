# True-eigenvector and higher-power probes

Supplement to the frozen 32-pair study `fresh32_std003_4800_20261126_010609`. No mother models were retrained.

Four additional targets (q32, q64, q128 and the true leading eigenvector), five depths including input, three probe splits and 64 frozen models produce 3,840 score records. Probes use 1,200 updates and validation-only selection. The upper/lower query-direction audit is inference only (1,920 records).

True-eigenvector learned-probe similarities by layers 1–4: 111 = 12.75%, 21.29%, 42.35%, 70.75%; 000 = 12.74%, 21.25%, 49.12%, 77.76%. The layer-3 paired difference is -6.77 percentage points (95% seed-bootstrap CI [-8.15, -5.38]). No clear normalization advantage appears in layers 1–2. Final task performance differs between groups, so this is not an accuracy-matched causal test of computation placement.

Raw coordinate readouts remain much lower than learned probes before layer 4; neither query half shows persistent monotonic alignment. Decodability does not establish standard power-iteration dynamics. Higher powers do not necessarily score worse: layer-4 scores rise from q32 toward the true target in both groups.

Independent calibration: mean squared cosine to truth is 77.68%, 90.84%, 97.02%, 99.08% for q16/q32/q64/q128. The metric is directional similarity, not classification accuracy.

## Files and reproducibility

Open `eigen-highpowers-report.html` locally for the Chinese report, paired confidence intervals, all-32/25-pair sensitivity toggle, high-power table, fixed readouts, shuffled controls and embedded CSV downloads. `report-statistics.json` contains report statistics. `build_eigen_highpowers_report.py` regenerates the report with NumPy and Matplotlib.

The release archive contains all new probe weights, validation histories, target arrays, source, executed notebook, protocol, audits and a per-file SHA-256 manifest. Exact parent input pools are preserved in the preceding probe1200 release; mother checkpoints are in the baseline release. All 64 checkpoint hashes were independently checked against the original downloaded archive. No 10%–99% convergence-tier probes were fitted in this supplement.

[Complete supplementary archive](https://github.com/tianrumao/power-iteration-normalization/releases/tag/eigen-highpowers-20261127)

Archive: `eigen-highpowers-probe1200-20261127.zip` (143,651,079 bytes). SHA-256: `2d09694afe1963bcf48bdc4a1d42be56955f7bf9a7172ff587676903dc6b4301`.
