# Compact forecast artifacts

Section 12 table scripts can read either the original forecast pickle or the
compact NPZ format. When both exist, the loader prefers the compact file.

From the repository root, run:

```bash
python "12. Evaluation of Probabilistic Forecasts/tables/compact_forecasts.py"
```

The converter scans for `*preds_test_set.pkl`, writes one
`compact_test_data.npz` containing fields that are identical across every
model, and writes a `*.compact.npz` next to each model pickle. It checks shared
fields by exact value equality. `true`, `u_grid`, `seq`, and feature-name/list
fields are candidates; fields that differ stay with their model. Model-specific
forecasts and diagnostics are preserved. The source pickle files are not
changed. Use `--force` to rebuild existing compact files.

The NPZ files use lossless ZIP compression and do not use Python-pickled object
arrays. The converter must be run in a Python environment that can read the
source pickles, since old pickle files can depend on the NumPy and Pandas
versions used when they were written.

To convert only selected artifacts:

```bash
python "12. Evaluation of Probabilistic Forecasts/tables/compact_forecasts.py" --inputs \
  benchmark_outputs/naive/preds_test_set.pkl \
  benchmark_outputs/garch_student_t/preds_test_set.pkl
```

The table loader finds the shared file by searching parent directories, so
model outputs can remain in their existing subdirectories.
If compact files already exist, rebuild the full source set together so they
continue to point at the same shared file.
