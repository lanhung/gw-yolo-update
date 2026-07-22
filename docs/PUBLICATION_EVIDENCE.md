# Publication evidence ledger

`publication-evidence-audit` is the fail-closed final aggregation layer for the GW-YOLO paper
workflow. It does not turn queued jobs, smoke tests or validation diagnostics into scientific
claims. Instead, it binds each predeclared gate to one JSON report, recomputes the report hash,
replays selected model/manifest hashes, evaluates declared predicates and writes both a
machine-readable audit and an optional Markdown table.

The validation-freeze protocol is
[`configs/publication_validation_evidence.yaml`](../configs/publication_validation_evidence.yaml).
It covers the source-safe aligned corpus, purpose-disjoint endpoint, five-seed model, paired
fixed-epoch/fixed-update scaling curve, continuous candidate calibration, paired raw/mask
`<VT>`, calibration perturbations, detector-set OOD/run transfer, matched-event within-backend
DINGO/AMPLFI PE portfolio and the
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

For the final validation-freeze handoff, use
`scripts/run_publication_validation_ledger.sh`. It requires one explicit path for each of the ten
declared reports, refuses missing or duplicate output paths, invokes `--require-ready`, and then
independently checks 10/10 passing states plus every artifact replay. The final requirement is an
already-frozen `locked_evaluation_corpus_unopened` contract; the ledger runner does not create that
contract or read O4b event-level data.

The continuous-search entry must be the output of
`candidate-search-calibration-endpoint-bind`, not a standalone threshold report. The binding
replays the 3,000-row purpose-disjoint endpoint, five-seed selection, candidate pipeline,
injection rankings, continuous block-permutation background and calibration. It also requires no
GPS-block overlap between threshold background and injection validation, exactly the predeclared
0.1/year FAR target and at least 10,000 bootstrap replicates.

The paired raw/mask entry must be the output of
`candidate-search-raw-mask-endpoint-bind`, not the standalone comparison JSON or unbound runner
receipt. The binder replays the source receipt, purpose-disjoint background authorization, parent
plan, both arm merges, both calibrations, mask validation/timing receipts and paired comparison.
This prevents an otherwise valid-looking comparison made on a pre-partition or
capacity-insufficient plan from entering the validation ledger while preserving the immutable
commit that executed the expensive background run.

The detector-set OOD entry must be the output of
`detector-set-ood-validation-bind`, not the inner embedding report or the runner receipt alone.
The binder replays the source-safe corpus audit, score-blind held-family protocol, leave-family-out
split, three exact split manifests, CUDA embedding report, checkpoint and both score manifests. It
also requires at least two observing-run strata, a known-only frozen threshold and an auxiliary
policy that cannot veto a strain-coherent candidate. This keeps the locked scorer's native
validation-model input separate while ensuring that only corpus-bound OOD evidence can satisfy the
validation ledger.

`publication_ready=true` in the validation protocol means only that the predeclared inputs needed
to freeze and authorize the one-time locked evaluation are present. It deliberately keeps
`scientific_claim_allowed=false`. A separate `locked_final` protocol, populated only after the
exclusive access receipt and locked search/PE reports exist, may set
`locked_final_evidence_complete=true`, but the ledger still keeps
`scientific_claim_allowed=false`. Only the immutable locked result reports and their statistical
interpretation can support a paper claim; the aggregation utility cannot authorize one.

The final protocol is now predeclared in
[`configs/publication_locked_final_evidence.yaml`](../configs/publication_locked_final_evidence.yaml).
It requires nine artifacts: the frozen suite plan, exclusive access receipt, raw and mask search
arms, paired fixed-FAR `<VT>` result, OOD transfer, locked DINGO/AMPLFI within-backend PE portfolio,
catalog diagnostic and
the final all-output hash receipt. A non-significant or negative paired result is still a valid
completed endpoint and is retained; it cannot be replaced by a new threshold or omitted from the
ledger.

## Omission-resistant paper registry

`publication-result-registry` converts one validation ledger and, after the one-time evaluation,
one locked-final ledger into immutable JSON, CSV and Markdown outputs. It does not trust an old
ledger at face value: it reloads the exact hashed protocol config, reevaluates every predicate,
rehashes every bound evidence report and replays every declared artifact. The live `path_absent`
check is also reevaluated, so a pre-access validation ledger cannot be presented as current after
the access log has appeared.

Every protocol requirement is exported, including `pending`, `failed`, skipped, non-significant
and negative rows. For reports that expose `primary_endpoint_result`, `promote_to_paper`,
`promotion_checks`, `endpoint_outcomes` or endpoint-completion fields, the raw outcome is retained
in the JSON and CSV rather than reduced to a favorable boolean. A non-significant primary mask
result is labeled `gate_passed_null_or_negative_primary_endpoint`; it is never filtered out.

Create the validation-stage registry without opening or naming a locked result:

```bash
python -m gwyolo.cli publication-result-registry \
  --ledger /artifacts/publication-validation-ledger.json \
  --output /artifacts/publication-result-registry-validation.json \
  --csv /artifacts/publication-result-registry-validation.csv \
  --markdown /artifacts/publication-result-registry-validation.md
```

After the one-time locked evaluation has completed, use new output paths and add exactly one
`--ledger /artifacts/publication-locked-final-ledger.json`. Duplicate phases, changed evidence,
omitted requirements and preexisting output paths fail explicitly. The registry always writes
`scientific_claim_allowed=false`: it is an artifact-retention and paper-table index, not a claim
promotion mechanism.
