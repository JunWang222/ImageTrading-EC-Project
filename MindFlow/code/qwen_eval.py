"""Run the Step 9a2 Qwen zero-shot pilot on paired v4 volatility windows."""

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

import final_model as fm
import indicator_cnn as ic
import pure_image as pi
import regime
from chartcnn import Store


MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
QUESTION = (
    "Predict whether the absolute return over the next five minutes exceeds "
    "this stock's median absolute five-minute return in the training period. "
    "Answer with a single letter: A for above the median, B otherwise."
)
LEGEND = {
    "6b": "The image shows OHLC, volume, and an exponentially weighted volatility band.",
    "6d": (
        "From top to bottom, the image shows OHLC and volume with a gray session-start "
        "marker, stock volatility, SPY volatility, relative volume, and multi-day "
        "realized volatility. The bottom strip encodes standardized features as bars "
        "above or below zero."
    ),
}


def prepare_inputs(prepared, n_rows, scale_rows, seed):
    fm.register()
    store = Store(prepared)
    embargo = ic.embargo_dates(store, 2)
    train = ic.decision_windows(store, "train", 12, embargo, max_rows=scale_rows, seed=seed)
    valid = ic.decision_windows(store, "validation", 12, embargo, seed=seed)
    cell = {"target": "vol", "horizon": 1, "stride": 12}
    _, median = ic.cell_rows(store, cell, "train", train)
    labelled, _ = ic.cell_rows(store, cell, "validation", valid, median)
    # Spread the small pilot over the validation year without sampling on labels.
    days = np.sort(labelled.decision_date.unique())
    chosen = days[np.linspace(0, len(days) - 1, min(n_rows, len(days)), dtype=int)]
    rows = (labelled[labelled.decision_date.isin(chosen)]
            .sample(frac=1, random_state=seed).drop_duplicates("decision_date")
            .sort_values("decision_date").reset_index(drop=True))
    if rows.empty:
        raise ValueError("No eligible validation windows")

    halflife = 39
    rv = regime.rv_tables(store, min_bars=60)
    _, raw_train = fm.numbers(store, train, rv)
    _, raw_valid = fm.numbers(store, rows, rv)
    scaler = ic.TabScaler(raw_train)
    ewsd = ic.fit_ewsd_scale(prepared, train, halflife, n=len(train), seed=seed)
    scales = pi.fit_scales(prepared, train, pi.log_rv(raw_train), store.market_close,
                           halflife, ewsd, n=len(train), seed=seed)
    charts = {
        "6b": ic.Charts(prepared, rows, "ewsd_tab_rv", ewsd, halflife),
        "6d": pi.PureCharts(prepared, rows, "pure_all", scales, halflife,
                            store.market_close, pi.log_rv(raw_valid), scaler(raw_valid)),
    }
    rows["bar_of_day"] = rows.end_index % ic.LOOKBACK
    rows["abs_return"] = rows.forward_return.abs()
    rows["train_abs_return_median"] = rows.security_index.map(median)
    return rows, charts, raw_valid, {"ewsd": ewsd, "strips": scales, "tab": scaler.state()}


def answer_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    if any(len(tokens) != 1 for tokens in ids) or ids[0] == ids[1]:
        raise ValueError(f"Expected distinct single-token answers, got {ids}")
    return [tokens[0] for tokens in ids]


@torch.inference_mode()
def score(model, processor, image, prompt, ids):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    batch = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    logits = model(**batch, use_cache=False).logits[0, -1].float()
    yes, no = logits[ids].tolist()
    probability = float(torch.sigmoid(torch.tensor(yes - no)))
    return yes, no, probability, batch["image_grid_thw"][0].tolist()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(prepared, bundle, output_root, n_rows=32, scale_rows=2000, seed=69, revision="main"):
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Run with a GPU, e.g. Singularity exec --nv")
    if not 1 <= n_rows <= 247 or scale_rows < 2:
        raise ValueError("Use 1–247 validation days and at least two training scale windows")
    repo = Path(__file__).resolve().parents[2]
    run_id = datetime.now(timezone.utc).strftime("qwen_zero_shot_%Y%m%d_%H%M%S_%f")
    output = output_root / run_id
    output.mkdir(parents=True)
    print("Loading Qwen on", torch.cuda.get_device_name(0), flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    ids = answer_ids(processor.tokenizer)
    print("Building volatility labels and train-only preprocessing", flush=True)
    rows, charts, numbers, prep = prepare_inputs(prepared, n_rows, scale_rows, seed)
    settings = {
        "mode": "zero_shot", "model": MODEL_ID, "requested_revision": revision,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
        "bundle": bundle.name, "bundle_sha256": sha256(bundle),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "seed": seed,
        "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
        "requested_rows": n_rows, "validation_rows": len(rows), "scale_rows": scale_rows,
        "target": "vol_h1", "stride": 12, "embargo_sessions": 2, "halflife": 39,
        "min_bars": 60, "question": QUESTION, "legend": LEGEND, "answer_token_ids": ids,
        "feature_order": numbers.columns.tolist(),
        "image_processor_class": type(processor.image_processor).__name__,
        "image_processor": processor.image_processor.to_dict(),
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    (output / "preprocessing.json").write_text(json.dumps(prep, indent=2) + "\n")
    pd.DataFrame([{"model": MODEL_ID, "total": sum(p.numel() for p in model.parameters()),
                   "trainable": 0}]).to_csv(output / "network_parameters.csv", index=False)
    pred = rows[["security_index", "end_index", "decision_date", "bar_of_day",
                 "label", "abs_return", "train_abs_return_median"]].copy()
    summaries = []
    for style, chart in charts.items():
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        scores = []
        for i in range(len(rows)):
            prompt = LEGEND[style]
            if style == "6b":
                values = ", ".join(
                    f"{name}={value:.8g}" if np.isfinite(value) else f"{name}=missing"
                    for name, value in numbers.iloc[i].items()
                )
                prompt += "\nHistorical features: " + values
            prompt += "\n" + QUESTION
            image = Image.fromarray(chart.image(i)).convert("RGB")
            yes, no, prob, grid = score(model, processor, image, prompt, ids)
            scores.append((yes, no, prob, *grid))
            if i == 0 or (i + 1) % 8 == 0:
                print(f"{style}: {i + 1}/{len(rows)} scored, source image={image.size}, grid={grid}", flush=True)
        score_frame = pd.DataFrame(scores, columns=[
            "logit_yes", "logit_no", "prob_yes", "grid_t", "grid_h", "grid_w"])
        for column in score_frame:
            pred[f"{style}_{column}"] = score_frame[column]
        pred.to_parquet(output / "predictions.parquet", index=False)
        y, p = pred.label, score_frame.prob_yes
        summaries.append({
            "input_style": style, "rows": len(y),
            "auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
    table = pd.DataFrame(summaries)
    table.to_csv(output / "summary.csv", index=False)
    (output / "takeaways.md").write_text(
        "# Qwen zero-shot pilot\n\n"
        "Both input styles share the same validation keys. Preprocessing was fitted "
        "on sampled training windows. These pilot scores establish execution; paired "
        "comparison with saved CNN predictions and held-out testing are separate runs.\n"
    )
    print(table.to_string(index=False))
    print("Saved:", output)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--scale-rows", type=int, default=2000)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    run(args.prepared, args.bundle, args.output, args.rows, args.scale_rows, revision=args.revision)


if __name__ == "__main__":
    main()
