# TODO

Goal: turn the frozen SLICES thesis benchmark into a small, auditable public
research artifact. Keep future scientific extensions out of this file.

## P0 - Public Evidence Package

- [ ] Publish the thesis artifact release.
  - Include final thesis PDF, `results/slices-thesis-v1/`, diagnostic result packages, `REPRODUCIBILITY.md`, `repro.lock`, and expected hashes.
  - Keep raw ICU data private. Include prediction-level metric inputs only if privacy/legal review allows it; otherwise document the limitation.

- [ ] Add release metadata.
  - [x] Add `CITATION.cff`.
  - [x] Document the public equivalent of launch commit `ffd6dc52c2ed224da5290cfb5d1a6634b5ba6ab6` (tag `thesis-v1-launch` on `d1b86f6`; README and REPRODUCIBILITY).
  - [ ] Push tag `thesis-v1-launch`; add a changelog entry.
  - [ ] Replace mutable thesis, README, and reproducibility links with the immutable release URL or DOI once published.
  - [ ] Fill in the PMLR volume in `CITATION.cff` and the README BibTeX.

- [ ] Publish a cleaned evidence ledger after thesis finalization.
  - Add `CLAIMS.md` or `EVIDENCE.md` only after claims are final.
  - Include artifact filters, claim tiers, statistical boundaries, checkpoint caveats, JEPA caveats, fairness caveats, and privacy/access limits.

## P0 - Scope Hygiene

- [x] Remove SMART from public defaults.
  - Default launcher/export/fairness scope excludes `smart_external_reference`.
  - Default corpus count is `2455`, not `2590`.
  - README/docs state SMART is optional external reference code, not part of the controlled thesis corpus.

- [ ] Reconcile launch, export, and evidence counts.
  - [x] Document launched runs vs evaluated rows (REPRODUCIBILITY: 2455 = 145 pretraining + 2310 downstream).
  - [ ] Validate expected seeds, revisions, launch commit, and package row counts in `scripts/audit_results.py`.

## P0 - Reproducibility Surface

- [x] Add a public results audit: `uv run python scripts/audit_results.py`.
  - Validates manifest hashes, row/column counts, unlisted files, and path leaks in text and parquet strings without W&B, raw ICU data, or encoder weights.
  - [ ] Table/figure rendering still lives in the private thesis repository.

- [ ] Add public CPU smoke path.
  - Provide a tiny synthetic RICU-shaped fixture and one command that runs preprocessing, one-epoch MAE pretraining, one-epoch finetuning, export, and validation.
  - Mark synthetic metrics as functional checks only.

- [x] Document the credentialed full rerun path (REPRODUCIBILITY section 3).
  - [ ] Add a rerun-vs-frozen comparison tool with documented tolerances.
  - [ ] Export the 145 pretraining run configs and link each downstream row to its pretraining run ID and encoder SHA256.

- [ ] Add `repro.lock`.
  - Record public commit, launch commit/public-equivalent mapping, `uv.lock` hash, Python/R/RICU versions, result hashes, row counts, schema versions, and expected seeds.

## P0 - Automated Guards

- [x] Validate public result manifests in CI.

- [ ] Add schema-versioned result contracts.
  - Fail when public parquet schemas, row counts, or required provenance fields drift without migration notes.

- [ ] Store release-grade checkpoint provenance.
  - For each neural row, record checkpoint source, encoder/config/eval artifact hashes, and whether weights are public, private, or unavailable.

- [ ] Make fairness auditability explicit.
  - Store subgroup counts, valid groups, metric denominators, and provenance columns where shareable.
  - State that fairness results are descriptive unless statistical tests are added.

## Definition Of Done

- [ ] Fresh clone can run public audit and verify all committed result packages.
- [ ] Fresh clone can run CPU synthetic smoke without W&B or ICU credentials.
- [ ] Full rerun path is documented and clearly marked credentialed/private.
- [ ] Every public result claim has a file, filter, metric, seed count, hash, and caveat.
- [ ] README, release notes, thesis PDF, and public evidence package make the same claims.
- [ ] SMART, TS2Vec, supervised, GRU-D, XGBoost, and fairness are scoped consistently.
- [ ] The public tag or patch-equivalence check ties the release to the scientific code state.
