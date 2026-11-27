# Probe budget and target-definition audit

Follow-up to the preserved 300-step baseline. All 64 mother models stay frozen. All 7,680 probe scores were refit with a 1,200-update budget using the original split seeds, targets, initializations, loss and validation-only selection.

## Findings

The original intermediate-layer probes often selected the 300-step endpoint (1,942/4,608 maps). After extension, 421/4,608 select the 1,200-step endpoint. The mean extra validation gain over steps 900–1,200 is 0.113 percentage points; the 95th percentile is 0.411 points. This is a budget sensitivity check, not a proof of global optimality.

On identical probe test examples, the unchanged final model output scores 70.68% / 77.76% against the true leading eigenvector and 61.71% / 66.72% against q16 (111 / 000). The refitted q16 probes score 61.69% / 66.72%. Thus the main discrepancy comes from different target directions, not from the probe failing to recover the original readout. Validation selection can produce tiny test-score changes.

## Preserved artifacts

The corresponding release contains all fitted probe weights, validation histories, partitions, comparison tables, per-model checkpoint/data audit, standalone scripts and 32 verified probe input/label pools. Regenerated probe inputs and power labels match the saved parent SHA-256 fingerprints exactly; eigenvector references were recomputed and are identified as such. Original mother-model training arrays were not regenerated.

The parent release retains all mother-model checkpoints and original run files: https://github.com/tianrumao/power-iteration-normalization/releases/tag/fresh32-4800-20261126

Read probe-convergence-report.html for the complete target and convergence comparison.
