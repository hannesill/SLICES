# Final versus best-SSL-validation checkpoint diagnostic

This is a separate post-hoc analysis of MIMIC-IV hospital mortality and remaining
LOS using frozen linear probes. It contains 60 CPU probe runs, paired across the
two checkpoint policies, three core objectives, two tasks and five seeds. The
final arm was rerun in the same cached CPU implementation as the best-validation
arm; it was not replaced with the frozen historical scores.

Run the standard-library audit from the supplement root:

```sh
uv run --no-project python audit.py
```

`audit-output/checkpoint-analysis.json` verifies the exact grid, 30 paired
differences, six cell summaries, all 60 validation selections including earliest
ties, labelled support counts, and differences from the historical final scores.
`uv run --project framework python render.py --out paper/generated` regenerates
the checkpoint table along with the original paper assets.

The runs used the same patient splits and train-only normalizer within each pair,
reset head and batch RNG, a frozen 159,808-parameter encoder, linear heads, AdamW
at 1e-4 with weight decay .05, cosine decay, batch size 64, at most 50 epochs,
validation patience 10, and gradient clipping at 1. Classification used
square-root-balanced class weights and label smoothing .1. LOS used squared-error
training with absolute-error selection/reporting. Validation selected the head;
test labels entered evaluation once afterward.

MAE led hospital mortality and contrastive led LOS under both policies in all
five seeds. JEPA's best-minus-final changes averaged -0.0406 AUPRC and +0.2638 LOS
days, worsening in all five seeds on each task. Its LOS mean rank moved from
second to third. These are scoped descriptive observations, not checkpoint
equivalence or a claim about other databases, tasks, or fine-tuning.

Rerun-final differs from frozen-final. Cell-mean differences range from -0.001018
to -0.000106 AUPRC and -0.003616 to +0.010286 LOS days. The largest absolute
per-seed LOS deviation is 0.090066 days. The CPU cached execution and dedicated
paired RNG streams differ from historical GPU/Lightning execution. Both policies
are compared within the diagnostic, never across execution implementations.

`results/checkpoint-sensitivity/` retains numeric CSVs, aggregate validation
histories, class/support counts, the protocol, runtime ledger and exact executed
source. Producer receipts report verification of 30 archived encoder hashes and
83 source/evidence hashes. The anonymous projection removes account/run
identities, private split/label/normalization fingerprints, and operator paths.
Public audit verifies arithmetic and retained metadata; it cannot reopen private
inputs or independently verify their identity. No records, embeddings, or
trained patient-data weights are supplied.

The producer's separately authorized campaign took 1,175.752 seconds of worker
wall time on CPU, including its runtime preflight, with no failed or timed-out workers. This time
is separate from the extension's synthetic validation budget. Clinical checkpoint
retraining cannot be run from the anonymous ZIP alone: it requires authorized
processed inputs, the archived encoders and original frozen exports. The retained
execution source documents that procedure; no audit or rendering command starts
it. The unresolved author ethics confirmation in the manuscript remains open.
