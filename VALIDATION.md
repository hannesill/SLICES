# Extension validation record

This is development-agent validation on macOS arm64, not independent human reuse.
No person outside the development workflow has completed a reuse study. A fresh
extraction and fresh virtual environment test packaging and instructions on this
host; they do not establish cross-platform support or clinical replication.

The recorded acceptance run followed the packaged README, TS2Vec case, and
training guide. It copied no virtual environment, source, or configuration from
another checkout. uv reused its interpreter/distribution caches. Python was
3.12.12, PyTorch 2.9.1, and Lightning 2.6.0. The recorded source hashes bind these
checks to their payload; the enclosing ZIP may also include subsequent manuscript
or separately identified analysis updates.

| Documented step | Wall seconds | Result |
| --- | ---: | --- |
| Standard-library aggregate audit | 0.138 | Pass |
| Fresh locked installation, 92 packages | 1.347 | Pass |
| Frozen extension report | 0.119 | Pass |
| Complete synthetic extension, including imports | 36.826 | Pass, four downstream runs |
| Five-objective synthetic smoke | 3.201 | Pass |
| Figure/table rendering | 2.751 | Pass, all 11 source assets byte-identical |
| Four printed TS2Vec training configurations, composition only | 11.516 | Pass, no clinical training |
| Preprocessing dry-run | 0.061 | Pass, no data read |
| Extracted manuscript compilation | 4.943 | Pass, eight main pages |

`validation/instructions.json` contains command arguments, timings, limits, host
type, archive identity and instruction hashes. `validation/extension-case.json`
records transfer, update and test-selection checks plus executed source/config/
checkpoint hashes. `validation/frozen-extension.json` records the 24 scientific
cells separately; test-support counts are absent for all 120 TS2Vec rows and
remain unknown. The final paper package has a separate author-facing validation
receipt binding its exact archive and PDF, including visual inspection.

## Failures and assistance before acceptance

1. The base recipe printer rejected `--objective ts2vec`. The printer now accepts
   the extension, marks its experiment class separately, and prints both protocols.
2. The first development run failed after training the first synthetic head:
   Lightning's weights-only checkpoint loader rejected its saved OmegaConf
   configuration. The runner now explicitly deserializes only the trusted
   checkpoint created earlier in that same process. That failed attempt took
   37.648 seconds including initial imports. The corrected development run took
   5.616 seconds and passed.
3. An added support-count test initially assumed a number where the frozen
   TS2Vec export has null. This exposed a reporting error: two nulls had been
   counted as a match. The report now records unknown support and tests differing
   known counts separately. All 29 targeted tests then passed in 2.02 seconds.
4. The local sandbox initially denied uv's shared cache access. Normal local
   execution allowed the fresh installation. One initial orchestration command
   also used the wrong working directory and failed before installation; it is
   not attributed to the package. Earlier runs inherited a virtual-environment
   variable that uv explicitly ignored. The acceptance runner removes inherited
   Python environment overrides before starting the documented commands.

These fixes used repository knowledge and agent assistance. The subsequent
recorded acceptance run used only packaged commands, with no source edits,
undocumented fallback, credentials, or clinical-data access. Library warnings
about low DataLoader worker counts, an available but unused accelerator, and
intentionally frozen modules are expected in the bounded CPU fixture.

To repeat the acceptance route in another fresh directory, use the included
standard-library `validate_instructions.py` with `--archive`, `--out`, and an
optional `--synthetic-budget-seconds` cap. It extracts the ZIP, checks that each
command occurs in the guides, records failures without applying fallbacks, and
kills the command's process group on timeout. It composes credentialed recipes
with `--cfg job`; it never launches clinical training. Inspect the complete PDF
visually after it completes.
