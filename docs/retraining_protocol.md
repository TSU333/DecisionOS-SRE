# Retraining protocol (MVP)

The original 15-case test has already been inspected. It is a regression set for this iteration, not an unseen confirmatory test. No regression, calibration, or gate labels select configurations/checkpoints. The existing split and data hashes remain unchanged.

1. Diagnose fitting on two TRAIN examples per fault (10 total). These weights are diagnostic only and never promoted.
2. Train the existing architecture with separate head/backbone learning rates, more optimizer updates, and zero whole-evidence dropout. Keep candidate shuffling and light metric dropout. Select checkpoints by validation sum of per-head NLL.
3. A numerical ExtraTrees control fits TRAIN only. Its metric-name vocabulary is TRAIN-derived; its input is the same causal summaries. It sees all summaries (the old text model can truncate), so it diagnoses available signal and is not an identical-input architectural comparison.
4. If validation warrants a representation change, declare it before evaluating regression. Any new serializer has an explicit version and checkpoint binding. Canonical ordering/aliases can be based on observable metric signatures, never answers.
5. Fit temperature only on calibration, policy only on gate_selection. Keep empirical joint-error target 5% and minimum 30 accepted independent runs. Additional training windows do not increase independent sample count.
6. Freeze the model/configuration choice before opening regression predictions. Once opened, no further tuning in this iteration. Run paired robustness checks, CPU benchmark, reload, and real HTTP verification.

Training on the same original 50 TRAIN runs does not create new independent evidence. Small validation/calibration/test splits limit claims. No LLM, distillation, ONNX, INT8, large model, paid API, or remediation.

References: https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/modernbert and https://docs.pytorch.org/docs/2.7/optim.html . ModernBERT documents cls/mean pooling and limited CLS visibility in local attention layers; mean pooling is a testable option, not a guaranteed improvement.

## Declared representation trials

Three SFT configurations start independently from the same pinned ModernBERT-base weights and seed 42. Each has at most 24 epochs / 312 optimizer updates, validation patience 6, batch size 1, gradient accumulation 4, backbone LR 2e-5, head LR 3e-4, 5% warmup and linear decay. We fit/compare a fixed small set, not a Cartesian hyperparameter search.

- budget: original v1 text and CLS pooling; training changes only.
- canonical: metrics-canonical-v2 and attention-masked mean pooling. Observable telemetry signatures determine service aliases and candidate ordering. Actual service identities are returned through the ID mapping. Descriptions/names are intentionally excluded from this metrics-only representation. Exact telemetry-signature ties use ID for deterministic tie-breaking, so identity invariance for indistinguishable services is not claimed.
- hybrid: identical canonical text/pooling with additional learned residual numerical MLPs for root/fault logits. This is an explicitly changed architecture, still with one shared encoder call. Only token-retained telemetry enters the numerical features. Metric-name vocabulary is learned from TRAIN; signed-log transforms and aggregation are fixed formulas with no fitted holdout statistics.

Numeric features per metric are signed log(1+min(abs(x),1e6))/5 for z, relative mean change, observed mean, baseline mean, plus missing fraction and observation-present flag. Relative change divides by max(abs(baseline),1e-6). Missing metric blocks are zeros with presence=0. Global per-column min/max/mean pools retained services in a canonical value order. It does not access gold-selected service features. Root scoring remains shared across candidates. These additions are controlled by config and included in pipeline binding.

The representation audit reads only 50 TRAIN + 15 model_validation cases. Full test evaluation waits until the selection JSON is written. Original artifact weights/bindings are verified unchanged. The diagnostic checkpoint is excluded from selection.

A fourth comparison freezes the same backbone with the hybrid representation/heads, head LR 1e-3, same maximum update budget and validation stopping. It is eligible for selection along with the old SFT and the three new SFT runs; minimum validation sum NLL remains the sole selection rule. This distinguishes benefits of numerical features/head training from full backbone fine-tuning.
