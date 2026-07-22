# Publication evidence ledger

`publication-evidence-audit` is the fail-closed final aggregation layer for the GW-YOLO paper
workflow. It does not turn queued jobs, smoke tests or validation diagnostics into scientific
claims. Instead, it binds each predeclared gate to one JSON report, recomputes the report hash,
replays selected model/manifest hashes, evaluates declared predicates and writes both a
machine-readable audit and an optional Markdown table.

The current validation-freeze protocol is
[`configs/publication_validation_evidence.yaml`](../configs/publication_validation_evidence.yaml).
It covers the source-safe aligned corpus, purpose-disjoint endpoint, five-seed model, paired
fixed-epoch/fixed-update scaling curve, continuous candidate calibration, paired raw/mask
`<VT>`, calibration perturbations, detector-set OOD/run transfer, paired DINGO/AMPLFI PE and the
still-unopened locked corpus. Missing bindings remain `pending`; malformed reports, failed
predicates and changed replay artifacts become `failed`.

Run an incremental audit without authorizing locked access:

```bash
python -m gwyolo.cli publication-evidence-audit \
  --config configs/publication_validation_evidence.yaml \
  --evidence source_safe_corpus=/artifacts/gravityspy_corpus_audit.json \
  --evidence independent_validation_endpoint=/artifacts/independent_endpoint.json \
  --evidence five_seed_model=/artifacts/five_seed_overlap_summary.json \
  --output /artifacts/publication-readiness-<commit>.json \
  --markdown /artifacts/publication-readiness-<commit>.md
```

Add `--require-ready` only at the validation-freeze boundary. The report is written atomically
before a non-ready audit exits nonzero, so failures are retained. Outputs are immutable and must
use a new path for a later snapshot.

`publication_ready=true` in the validation protocol means only that the predeclared inputs needed
to freeze and authorize the one-time locked evaluation are present. It deliberately keeps
`scientific_claim_allowed=false`. A separate `locked_final` protocol, populated only after the
exclusive access receipt and locked search/PE reports exist, may set
`locked_final_evidence_complete=true`, but the ledger still keeps
`scientific_claim_allowed=false`. Only the immutable locked result reports and their statistical
interpretation can support a paper claim; the aggregation utility cannot authorize one.
