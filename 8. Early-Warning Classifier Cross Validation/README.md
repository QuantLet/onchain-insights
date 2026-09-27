<p align="center">
  <a href="https://quantlet.com">
    <img
      src="https://github.com/StefanGam/test-repo/blob/main/quantlet_design.png?raw=true"
      alt="Header Image"
      width="100%"
    />
  </a>
</p>

```
Name of Quantlet: Onchain Insights - Early Warning Classifier Cross Validation

Published in: Onchain Insights - Early Warning Classifier Cross Validation

Description: We train early warning models for depeg detection. Depeg is defined as an absolute deviation of the pool price above a given threshold. We apply 5-fold expanding window cross validation in order to compare the performance of common tree-based architectures. The dataset is comprised of features generated in previous quantlets describing onchain liqudiity conditions and broader market state. In particular we show the model performance for varying $\alpha$ tuning the Gegenbauer basis for the decomposition of the Uniswap USDC-USDT liquidity curve.

Keywords: Cryptocurrency, Blockchain, Stablecoins, Decentralized Finance, Liquidity, Depeg risk

Author: Owen Chaffard

Submitted: 04.05.2026

Datafile: ./data/*
```

# Running the code

- Update data for all the repo. This shell script downloads the last release from the daily updated dataset and updates the corresponding files in all quantlets.

```bash
bash update_data.sh
```

- Move terminal into this specific quantlet:

```bash
cd '8. Early-Warning Classifier Cross Validation/'
```

- Activate the VM environment containing the tree-model dependencies and, for
  the foundation benchmark, `tabpfn` and `causilo`.

- The first shell script runs full 5-fold cross-validation for all models and automatically updates the preprocessed datasets:

```bash
bash compare_model_cv_15bp.sh
```

- This shell script also runs sensitivity analysis on the depeg threshold (defaults to 5 10 15 25 bps):

```bash
bash compare_model_cv.sh
```

- You can update the full retraining shell script with the best performing model in CV. The full retraining script then does last training plus the full suite of plots, including SHAP explanations:

```bash
bash full_retraining.sh
```

## Tabular foundation-model benchmark

Run [TabPFN 3.5](https://github.com/PriorLabs/TabPFN) and
[Causilo](https://github.com/nums-ai/causilo) on the same six 15-bp datasets and
the same five-fold operational CV protocol as the tree models:

```bash
bash compare_foundation_model_cv_15bp.sh
```

Both models are pinned to `cuda:0` by default. The runner checks that PyTorch
can allocate on the selected GPU and stops before CV if CUDA is unavailable;
it never silently switches to CPU. Use `--tabpfn_device cuda:1` and
`--causilo_device cuda:1` to select another visible GPU. The verified GPU name
and device are recorded in each run's `hparams.json`.

The default dataset directory is `../9. SHAP explanations of Early Warning Model/preprocessed_datasets`;
override it with `--dataset_dir` if your VM stores those Parquet files elsewhere.
The first fit may need access to each model's checkpoint. To try one alpha and
one model first:

```bash
bash compare_foundation_model_cv_15bp.sh --alphas 0.3 --model_names tabpfn_3_5 --n_bootstrap 50
```

For reported results, run the full defaults (`n_bootstrap=1000`).
TabPFN now uses `--tabpfn_train_chunk_size 4096` by default. Within each
training fold, it forms disjoint, target-stratified row chunks from fit rows
only and assigns them as TabPFN ensemble contexts. Every fit row is included,
and chunks are repeated equally when needed to reach at least eight ensemble
members (`--tabpfn_chunk_min_estimators`). The validation and test rows are
never included. No single member sees the entire training history, so this
is a **different model specification** from full-context TabPFN; report the
chunk size and do not pool its results with an unchunked run. Use
`--tabpfn_train_chunk_size 0` to disable this strategy. Causilo is unchanged.
The realized chunk sizes and ensemble-member counts are recorded per fold in
`cv/fold_metrics.csv` and announced in `artifacts/cv_progress.log`.

`--predict_batch_size` starts at 64 rows by default. On a CUDA out-of-memory
error, the runner retries the same rows at half the batch size, down to one
row, and keeps the smaller size for subsequent predictions. This does not
change fit rows or CV folds, and all prediction rows are retained in order.
Record the realized batch sizes from `cv_progress.log` when reporting a run;
as with other inference settings, check numerical reproducibility if batch
size changes. If even one prediction row fails, the training context may be
too large for the available GPU memory; try a smaller TabPFN training chunk,
free memory held by other GPU processes, or use a larger GPU.

The terminal now reports every prediction batch, CV phase, and an elapsed-time
heartbeat every 60 seconds during slow fits or scoring. Each model run also
writes `artifacts/cv_progress.log` under its `lightning_logs` directory. Use
`--progress_every_batches 5` for fewer batch messages or
`--progress_interval_seconds 30` for more frequent heartbeats.

Open `foundation_vs_tree_cv.ipynb` after both experiments finish. It loads the
saved summaries, verifies matching fold boundaries and evaluation settings,
and exports a comparison table and figure. The notebook's within-model CIs
are not paired confidence intervals for a foundation-versus-tree difference.

# Generated plots

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/plots_summary/threshold_15bps/heatmap_cv_event_utility.png"
    alt="Operational Utility Heatmap"
  />
</div>

<p align="center">
  <b>5-fold cross validation results</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/roc_pr.png"
    alt="Final retraining ROC and PR curves"
    width="100%"
  />
</div>

<p align="center">
  <b>Final retraining AUC/AUPRC</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/timeseries/predictions_over_time.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>24-hour ahead predicted depeg probability out-of-sample</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp_full_retraining/catboost_threshold_15_alpha_0.3_fullfeatures_threshold_specific_event_utility_then_timely_recall_then_false_alert_burden/artifacts/plots/event_level_recall_and_lead_time.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Event-level recall and alert lead time</b>
</p>

<br>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_realised_depeg_events.png"
    alt="Predictions over time"
    width="100%"
  />
</div>

<p align="center">
  <b>Unique depeg events timeline</b>
</p>

<div align="center">
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_chronological_cv_splits.png"
    alt="Predictions over time"
    width="49%"
  />
  <img
    src="https://raw.githubusercontent.com/QuantLet/onchain-insights/main/8. Early-Warning Classifier Cross Validation/lightning_logs/cv_model_comparison_2026-09-23_15bp/paper_ready/figure_cv_events_by_fold.png"
    alt="Predictions over time"
    width="49%"
  />
</div>

<p align="center">
  <b>Cross Validation setup description</b>
</p>
