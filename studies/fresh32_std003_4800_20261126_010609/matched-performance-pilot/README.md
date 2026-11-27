# Matched-performance pilot and generalization audit

Two independent pilot seeds (9003, 9004), not a completed full-cohort experiment. The validation-defined matching gate passed after using condition-specific learning rates (111: .001; 000: .0005) and selecting checkpoints around the prespecified common 75% similarity level. The 78% candidate was not feasible for both pilot pairs under the candidate-band rule. No new linear probes were fitted.

Held-out test scores: seed 9003, 111 = 73.99%, 000 = 74.88% (gap 0.90 percentage points); seed 9004, 111 = 74.90%, 000 = 75.15% (gap 0.25 points). Both observed test mean gaps are below the primary 1-point tolerance; 2.5 points is the sensitivity tolerance. Test results were not used to select these checkpoints. These point estimates do not establish population equivalence.

Normalization train-minus-test gaps remain 2.75 and 3.08 percentage points. Matching final task performance does not establish absence of overfitting. The previous 12,800-step normalization endpoints scored about 76% on test but had larger 3.94–4.51-point train/test gaps. All successful and failed recipe results are preserved.

Open `matched-pilot-report.html` for the Chinese report, four-split scores, checkpoint steps, learning curves, endpoint-difference heatmap, generalization gaps and integrity checks. The first validation half (8,192 samples) selects checkpoints; the second half independently confirms matching. Training has 65,536 samples and test has 8,192.

No exact X-matrix duplication was found within or across the four splits. All audited coordinates in input/four-layer states on 128 held-out test examples per model were finite. These specific checks do not rule out all statistical overfitting or distribution shift. Interval limitations and one protocol-wording correction are documented in `protocol-clarification.json`.

The compact Colab archive contains all persisted weights, intermediate checkpoints, optimizer recovery files, histories, scripts, executed notebooks, exact data manifests and per-example evaluation scores. Its per-file manifest was verified locally. Actual raw .npy arrays remain in the persistent Drive dataset cache; their locations and digests are indexed in `RAW_DATA_INDEX.json` and they are not embedded in this compact archive.

The learning-rate repair plan was recorded before held-out test scores were available. It reuses the already trained 111 weights and does not retrain them. Both conditions retain identical initialization, data, batch sequence and schedule shape; learning rates and selected training steps differ explicitly.

A full-cohort rerun and new per-layer probes have not started. Generalization acceptance criteria and full-run data persistence still need to be finalized.

[Complete pilot archive](https://github.com/tianrumao/power-iteration-normalization/releases/tag/matched-pilot-20261127)
