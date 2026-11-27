# Dyadic power-target probe heatmaps, q1 through q256

The requested grid is q1, q2, q4, q8, q16, q32, q64, q128, q256 (2^0 through 2^8). No q512 probes were fitted. All 64 mother models remain frozen. Only q256 required new fitting: 960 new score records, with 1,200 updates per fit, three existing splits and validation-only selection. Earlier completed targets were reused, producing 8,640 combined score records. Each model is averaged across its three splits before aggregating training seeds.

Open `dyadic-probe-heatmaps.html` locally for the Chinese offline report. Its primary view restores numbered blue heatmaps with a common 0–100% scale. It includes the input baseline, all-32/25-pair sensitivity views, fixed-readout and shuffled-label controls, paired-difference heatmaps, highest-target tables and embedded CSV downloads. Standalone PNG/PDF heatmaps are supplied.

## Performance-matching limitation

This is an exploratory unmatched cohort. Same-example final fixed-readout similarities to truth are 70.68% (111) and 77.76% (000), a gap of about 7.08 percentage points. The old 0.1 flag threshold allowed 10 percentage points and is not a close-performance gate. Under the newly specified primary tolerance of 1 percentage point, only 2 existing pairs qualify; 4 qualify under the 2.5-point sensitivity tolerance. An independent matched-performance pilot is being run separately; no positive early-layer effect is required for acceptance.

## Data-generation diagnosis

All 32 saved probe pools (96,000 actual inputs) were checked without modification: zero padding, finite values, unit norms within float32 rounding, eigenvector references and q16 labels agree with independent spectral calculations. Actual-pool q16-to-truth similarity has mean 78.23%, median 94.10%, and 10th percentile 21.90%. The independent 2,048-example calibration also shows substantial skew. No numerical generation or label error was detected in these checks; convergence difficulty is heterogeneous. This audit does not claim to inspect the reclaimed original training-array cache.

Fixed k does not specify equal convergence progress across samples. Doubling k also does not yield uniform similarity percentages. A controlled-data extension should predefine spectral and initial-alignment strata. No new mother-model data distribution was substituted in this study.

For fixed X, the final eigenvector target is invariant to the initial vector a, while finite q_k(X,a) targets depend on a. Final-answer supervision does not require the learned network to reproduce this particular power-method trajectory. Heatmap maxima therefore do not identify exact algorithm steps. The report does not claim to have run a fixed-X/changed-a causal control.

## Reproducibility and archives

`dyadic-provenance.json` records every source score file and SHA-256. `dyadic-summary.csv` includes paired training-seed bootstrap intervals (20,000 resamples). Multi-target/layer comparisons are exploratory. The 25-pair subset uses the existing loose performance-gap threshold and is not strictly accuracy matched.

The complete release bundle contains the new Colab q256 archive (weights, validation histories, target arrays, source, protocol and executed notebook), merged results, report, figures, data-difficulty audits and analysis code. Its root `PUBLICATION_MANIFEST.json` records SHA-256 digests; the nested Colab archive retains its own per-file manifest. Existing q1–q16 weights and exact probe inputs remain in the `probe1200-20261126` release; q32–q128 and true-eigenvector weights remain in `eigen-highpowers-20261127`. Mother checkpoints remain in `fresh32-4800-20261126`.

[Full dyadic release](https://github.com/tianrumao/power-iteration-normalization/releases/tag/dyadic-q256-20261127)
