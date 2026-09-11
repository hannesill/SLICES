# SLICES code and reproducibility artifact

This supplement contains runnable source and **run-level aggregate metrics** for
the paper, plus separate post-hoc diagnostics. It contains no patient records,
patient identifiers, trained patient-data weights, data-access tokens, or credentials.

Download the repository ZIP or check out the submission branch. Run commands from
the directory containing this README. The artifact includes its own source,
configs, results, renderer, and paper. No sibling repository is required.

Shell commands require `uv` on PATH. Python 3.12+, a dependency cache or
internet for installation, and a Unix-like environment are needed for the CPU
examples. The audit also works with an existing Python 3.12+ interpreter without
uv. PDF compilation additionally requires `latexmk`, PDFLaTeX and BibTeX.

Start with the aggregate audit, then install the environment and run the CPU
smoke test. `RELEASE.json` identifies the packaged inputs. `MANIFEST.json` pins
every supplied file. A downloaded mirror should pass the audit before any files
are edited or regenerated.

| Path | Contents |
| --- | --- |
| `results/` | Run-level metrics, scientific configurations, and separate diagnostics |
| `framework/` | Maintained encoder, objective, training, and preprocessing source |
| `historical/` | Source and configs from the frozen experiment launch |
| `benchmark.json` | Fixed feature order, preprocessing, tasks, budgets, and evaluation rules |
| `paper/` | Matching manuscript source, figures, tables, and manual diagrams |
| `TRAINING.md` | Explicit recipes for credentialed reproduction |

## Audit the reported results without data access

Python 3.12+ is sufficient. No Python packages, account, or network is needed:

```sh
uv run --no-project python audit.py
```

Alternatively use an existing Python 3.12+ interpreter directly. The command checks
every packaged SHA256, the exact 420-row/84-cell core grid, each five-seed set,
and primary metric ranges. It recomputes means, sample SDs, paired differences,
label-fraction winner counts with explicit comparator sets, and the constant LOS
baseline. Output is `audit-output/analysis.json`. The source projection includes
all 2,310 frozen downstream/baseline runs. Each CSV row is one *training run*,
never one patient. `results/source_provenance.json` records frozen source hashes
and the separately sourced diagnostics. Anonymous projection strips accounts,
run IDs/URLs, local paths, and free-text configuration blobs, not metric values.

## Reproduce tables and figures

Install the locked source environment and the plotting extra, then render:

```sh
uv sync --project framework --frozen --extra viz --python 3.12
uv run --project framework python render.py --out paper/generated
```

This regenerates the numerical table fragments and the three paper figures from
the CSV. `paper/generated/report.json` records the computed values. No training or
patient-data access occurs. From `paper/`, the included LaTeX paper source builds
using `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`. Run the audit
before regeneration; different plotting-library versions may change figure bytes.

Five manual method diagrams are also supplied as SVG and vector PDF in
`paper/diagrams/`. Their source hashes and optional conversion instructions are
in that directory. They are separate from the numerical renderer; compiling the
paper uses the supplied PDFs and needs no diagram-conversion software.

## Exercise the framework on a CPU

```sh
uv run --project framework python smoke.py
```

The synthetic test has a 120-second timeout and one CPU thread. It takes an
optimization step through MAE, JEPA, contrastive, and the TS2Vec-style extension,
round-trips the shared encoder checkpoint, checks frozen-head and fine-tuning
gradient behavior, registers a toy objective, and rejects incompatible feature
order, training duration, and checkpoint declarations. It is a software check,
not a miniature reproduction of clinical performance.

See `EXTENDING.md` for the objective interface and `TRAINING.md` for credentialed
reproduction. Package installation may need internet; the aggregate audit does not.
The paper's ethics and data-access statements are in `paper/main.tex`.

`CHECKPOINT_CASE.md` describes a separately authorized 60-probe checkpoint
diagnostic. The same audit verifies its public aggregates and validation
histories; the renderer also generates its table. Private input identity remains
a producer receipt. `VALIDATION.md` records extension validation and assistance.

## Run a complete extension workflow

After the locked installation above:

```sh
uv run --project framework python extension.py --out extension-output
```

`TS2VEC_CASE.md` explains this shared-trainer case, its exact changed components,
fixed scientific contract, and synthetic departures. It runs TS2Vec-style
pretraining, v3 encoder export, both adaptation protocols for synthetic binary and
regression tasks, validation-selected testing, and separate reporting of the
existing 24 frozen extension cells. The process has a 120-second timeout and one
CPU thread. Passing this test is agent software validation, not independent human
reuse or clinical replication. Synthetic metrics never enter the frozen results.

The raw/mask clinical-data diagnostic is described in `results/raw_mask.json`.
Its exact executed source is `results/executed_raw_probe.py`, matched by SHA256.
For a new authorized local run, use `uv run --project framework python raw_probe.py
--data /path/to/processed/miiv --out /path/to/new-analysis`. The runner enforces a
900-second timeout, two CPU threads, and a fresh output directory; it saves only
run-level metrics and input hashes. Its checkout-provenance handling also works
outside Git. This rerunnable copy differs from the recorded source only in
formatting and optional Git metadata handling, not the experiment computation.

## Reproduction scope

The documented thesis matrix contains 2,455 launched runs, including TS2Vec as an
extension and excluding SMART. The export contains 2,310 evaluated downstream or
baseline runs; the other 145 runs are pretraining. The core comparison has 420
evaluated runs across 84 cells. SMART source remains an optional external
reference and is not part of those results. Use the explicit cell recipes in
`TRAINING.md`; this artifact does not include the development tmux launcher.

All comparisons are descriptive. Passing an audit checks the exported evidence.
Passing the synthetic workflows checks the software. Neither reproduces the
clinical training runs or establishes numerical retraining equivalence.

## License

Code uses the MIT license in `LICENSE`. The historical and framework copies also
retain their license. Bundled fonts retain their separate license in
`paper/diagrams/figma-conversion/`. Author identity is withheld for double-blind review.
