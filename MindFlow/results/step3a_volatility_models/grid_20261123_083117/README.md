# Step 3a, volatility target: grid_20261123_083117

Validation 2023 (247 days, 822,653 rows per cell), v4 bundle. Summary files only; per-row predictions stay on Drive.

History of this run:
- 2026-11-23: all networks and HGB on `vol_h1` (5 min) and `vol_h12` (60 min).
- 2026-11-23: ViTs retrained after the tokenizer fix (chartcnn 2026-11-23.2).
- 2026-11-25: VIXY panel added (`plain_spy_vixy`, `vit_spy_vixy`, chartcnn 2026-11-24.1) on `vol_h1` only
  (`HORIZONS = [1]`), so these tables now list the 5-minute cell only. The 60-minute cell is unchanged on Drive
  (`vol_h12/predictions.parquet`) and in Step 5's summary.
  CNN + SPY + VIXY: best epoch 2 (L4, bfloat16); ViT + SPY + VIXY: epochs 3-7 on a T4 (float32), best epoch 5.

Result (5 min): VIXY panel vs without, paired day bootstrap: tied for the CNN (0.661 vs 0.660) and the ViT
(0.663 vs 0.661). HGB hand + SPY 0.689 stays best.
