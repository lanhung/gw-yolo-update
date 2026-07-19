# GW-YOLO research agent instructions

## Mission

Build a reproducible, publication-grade gravitational-wave transient system that separates compact-binary chirps from detector glitches, preserves instance masks, and is evaluated on continuous detector data with injection-based sensitivity and false-alarm metrics.

## Non-negotiable scientific rules

1. Never split augmented images independently. Group by waveform/injection ID, glitch ID, GPS segment, detector, and observing run.
2. Never call catalog-image hit rate a search recall. Search claims require continuous background, injections, FAR/IFAR, and sensitive spacetime volume `<VT>`.
3. Never tune a threshold on the test set. Fit thresholds/calibration on validation data, freeze them, then evaluate once on the locked test set.
4. Preserve all chirp and glitch instances and segmentation masks. Do not silently keep only the highest-confidence chirp.
5. Keep detector identity and GPS metadata. H1/L1/V1 must be aligned before network-level fusion.
6. Use numeric time-frequency arrays for primary experiments. Rendered plots may only be a legacy baseline.
7. Prefer time-domain physical augmentation followed by a fresh transform. Image mosaic/mixup cannot be presented as a physical signal mixture.
8. Every result must identify code commit, config hash, data-manifest hash, model hash, seed, environment, and exact command.
9. Report uncertainty: paired bootstrap intervals for efficiency/`<VT>`, Wilson intervals for binomial rates, and calibration coverage for posterior products.
10. Do not claim superiority to AMPLFI or DINGO on detection mAP. Compare shared inference/latency/calibration tasks, and clearly state the distinct mask/deglitch advantage.
11. Never report augmented/rendered image count as physical sample count. Report unique waveform/injection, glitch, GPS block, IFO, and run counts.
12. Generated chirp+glitch mixtures require disjoint waveform IDs and disjoint glitch/GPS IDs across splits; splitting mixture rows alone is leakage.
13. Freeze a statistically useful evaluation corpus before using its results to choose the primary architecture.

## Repository conventions

- Source code lives under `src/gwyolo`.
- Experiment configuration lives under `configs`; no hard-coded machine paths in Python.
- Generated artifacts live under `artifacts` or a configured external output directory and are not committed.
- Tests live under `tests` and must run without a GPU.
- Use `python -m gwyolo.cli ...` as the stable command interface.
- Keep legacy reproduction separate from O4 publication experiments.
- New metrics must include a unit test with a hand-calculated example.

## Quality gates

Before a training result is accepted:

- data audit reports zero cross-split group overlap;
- all image-label pairs are valid;
- the configured seed and data manifest are saved;
- validation and test metrics are written as machine-readable JSON;
- test evaluation uses a checkpoint selected only by validation performance;
- the checkpoint is saved against the configured primary metric, not a framework-default fitness proxy;
- catalog predictions retain every instance and mask;
- failures are explicit and produce a non-zero exit status.

Before a paper claim is accepted:

- at least five seeds for learned baselines or a justified uncertainty protocol;
- O4b remains locked until model and threshold selection is complete on O1–O3/O4a;
- comparisons use the same injections, background, live time, waveform population, and FAR definition;
- negative and null results are retained in the experiment table.
- a group-safe data-scaling curve demonstrates whether the endpoint is data-, domain-, or representation-limited;
- rendered-image, physical-group, injection, and background-live-time counts are reported separately;
- O4 transfer is evaluated independently of in-domain mAP.

## Safety and remote execution

- Never overwrite `/root/GW-YOLO`, its weights, or its existing `runs` directory.
- Remote experiments must use a new directory such as `/root/GW-YOLO-v2`.
- Check for active package installations and GPU jobs before starting training.
- Do not terminate another process unless the user explicitly authorizes it.
- Use resumable run directories and atomic JSON writes.
