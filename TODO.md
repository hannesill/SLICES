# TODO

Goal: turn the frozen SLICES thesis benchmark into a small, auditable public
research artifact. Keep future scientific extensions out of this file.

## P0 - Public Evidence Package

- [ ] Publish the thesis artifact release.
  - Include final thesis PDF, `results/slices-thesis-v1/`, diagnostic result packages, `REPRODUCIBILITY.md`, `repro.lock`, and expected hashes.
  - Keep raw ICU data private. Include prediction-level metric inputs only if privacy/legal review allows it; otherwise document the limitation.

- [ ] Add release metadata.
  - Add `CITATION.cff`, changelog entries, and a public tag or documented public-equivalent commit for `ffd6dc52c2ed224da5290cfb5d1a6634b5ba6ab6`.
  - Replace mutable thesis, README, and reproducibility links with the immutable release URL or DOI once published.

- [ ] Publish a cleaned evidence ledger after thesis finalization.
  - Add `CLAIMS.md` or `EVIDENCE.md` only after claims are final.
  - Include artifact filters, claim tiers, statistical boundaries, checkpoint caveats, JEPA caveats, fairness caveats, and privacy/access limits.

## P0 - Scope Hygiene

- [ ] Remove SMART from public defaults.
  - Default launcher/export/fairness scope excludes `smart_external_reference`.
  - Default corpus count is `2455`, not `2590`.
  - README/docs state SMART is optional external reference code, not part of the controlled thesis corpus.

- [ ] Reconcile launch, export, and evidence counts.
  - Document launched runs vs evaluated rows vs aggregate rows.
  - Validate expected seeds, revisions, launch commit, and package row counts.

## P0 - Reproducibility Surface

- [ ] Add `uv run slices reproduce audit`.
  - Validate path-leak scan, manifest hashes, diagnostic checksums, table/figure render into `/tmp`, and source checks without W&B, raw ICU data, or encoder weights.

- [ ] Add public CPU smoke path.
  - Provide a tiny synthetic RICU-shaped fixture and one command that runs preprocessing, one-epoch MAE pretraining, one-epoch finetuning, export, and validation.
  - Mark synthetic metrics as functional checks only.

- [ ] Add `uv run slices reproduce final`.
  - Make this the documented credentialed path for full ICU extraction, W&B/local registry export, and thesis-scale reruns.

- [ ] Add `repro.lock`.
  - Record public commit, launch commit/public-equivalent mapping, `uv.lock` hash, Python/R/RICU versions, result hashes, row counts, schema versions, and expected seeds.

## P0 - Automated Guards

- [ ] Validate public result manifests in CI when large files are present.

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
