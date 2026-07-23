# Automatic mask policy

## Scientific boundary

The project does not assume that a real detector glitch has one uniquely correct hand-drawn pixel
boundary. Human annotations are optional disagreement diagnostics only. They are not merged into
training truth and do not gate publication or locked-data access.

Primary masks are generated from numeric physical components:

- software-injection chirp masks come from isolated waveform power;
- real-glitch masks come from isolated real-glitch strain using frozen per-IFO whitening, a fresh
  multi-Q transform and a fixed relative-power rule;
- model outputs remain soft probabilities;
- unknown or later-run artifacts may abstain through the frozen OOD policy.

The real-glitch masks are explicitly called pseudo-labels. They cannot support a standalone
pixel-accuracy claim.

## Frozen algorithm

For every available detector:

1. load and hash-check isolated `raw_glitch_strain`;
2. whiten that detector independently;
3. compute numeric multi-Q power at the configured Q values and frequency/time grid;
4. in every IFO/Q plane, retain pixels above the frozen fraction of that plane's maximum;
5. retain the explicit detector-availability mask and leave missing IFO planes zero.

Chirp masks use the same transform and threshold family on the isolated injected waveform after
per-IFO scale normalization. The exact configuration is
`configs/physical_overlap_factory.yaml`.

## Evidence and claims

`automatic-mask-policy-audit` recomputes every validation mask byte-for-byte. The publication
binder then joins this audit to the validation-only raw/mask continuous-background endpoint.

The primary evidence is:

- fixed-FAR efficiency and `<VT>`;
- clean-signal non-inferiority;
- timing and detector-set coherence;
- paired raw/contaminated/mask-conditioned PE coverage, bias, width and latency;
- OOD abstention and observing-run transfer.

Human-consensus IoU, automatic-mask IoU against human masks and absolute DINGO-versus-AMPLFI
ranking are not primary claims.
