# Pilot protocol

Generate independent train, validation, and test splits with unit Gaussian columns and a separate unit start vector. Use CPU float64 eigendecomposition and record the top eigengap. Keep the initial query and four power-step directions for evaluation. Train on the final eigendirection using sign-invariant squared-sine error.

Encode each data token as [x; 0] and the last token as [0; a]. Each of the four residual blocks has its own V and W. Norm-on applies unit L2 length to each token after blocks 1–3. Norm-off keeps the residual state as it is. Both readouts normalize the lower half of the last token. Zero and nonfinite readouts receive error 1 and a failure flag.

Within a pair, reuse the same splits, batch indices, initial parameter values, optimizer settings, device and step budget. Use validation to review the pilot. Set a learning threshold, failure-rate limit, and acceptable paired final-error gap before running the five new seed pairs. The default pilot evaluates the final checkpoint. Save failures and keep the held-out test for assessment after settings are selected. The evaluation command also supports checking the pilot plumbing on its separate pilot test split.

For each checkpoint, read out each block's state using the fixed readout. Then bypass one learned update at a time and preserve the residual input and its normalizer. In a second intervention, match the resulting token lengths to the intact model's outgoing lengths at that block. Continue through all later blocks. An impossible length match counts as a failure. Report the final error increase relative to the intact model.

Keep each seed's table, curves, environment, config, run status and checkpoint location. Use the same device and precision across the formal pairs. Discuss layer placement only when both conditions meet the pre-agreed criteria. Similar layer and power-step errors indicate similar progress toward the leading eigendirection. Verifying a literal power step would require a separate comparison of each update against the corresponding power-method transformation.
