# Fresh 32-pair power-method study

Completed run: `fresh32_std003_4800_20261126_010609`.

- 32 paired seeds (1001–1032), 64 models, conditions 111/000.
- Initialization SD 0.03; learning rate 0.0005; 4,800 updates per model.
- 7,680 direction-probe scores and 512 bypass scores.
- All seeds retained; 7 pairs have final validation-error gap > 0.1.

The associated repository release contains the full persisted Colab run: executed notebook, original source snapshots, config, histories, selected and recovery model checkpoints, 192 probe-weight archives, split indices, data manifests, logs and result tables. ARCHIVE_MANIFEST.json lists persisted files and SHA-256 digests.

## Raw-data availability

The previous Colab runtime was reclaimed before archiving. Its temporary input-array cache is no longer available. The generator, seeds, cache identities and data fingerprints are preserved. No regenerated arrays are presented as original data.

## Different targets

The backbone learns the true leading eigenvector of XX^T. The q16 probe targets normalize((XX^T)^16 a) on separate probe data. The two squared-cosine scores need not agree. Including the original readout in the probe supplies a baseline only against the same target on the same examples.

The HTML report covers all 32 pairs and the 25 pairs with validation gap <= 0.1. The latter subset is not strictly performance matched.
