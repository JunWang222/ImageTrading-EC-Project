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


def prepare_shared_validation(prepared, shared_keys, references, n_rows, seed):
    """Render the original CNN inputs and attach CNN scores on the same validation keys."""
    b = references / "step6b_final_model/final_20261125_222625"
    d = references / "step6d_pure_image/pure_20261127_023749"
    config = json.loads((b / "settings.json").read_text())
    settings = config["settings"]
    cell = next(c for c in config["cells"] if c["name"] == "vol_h1")
    keys = ["security_index", "end_index"]
    cols = keys + ["decision_date", "label"]
    rows = pd.read_parquet(shared_keys, columns=cols)
    if rows.empty or rows[cols].isna().any().any() or rows.duplicated(keys).any():
        raise ValueError("Shared validation keys must be nonempty, nonnull and unique")
    rows = rows.sample(n=min(n_rows, len(rows)), random_state=seed).sort_values(keys)
    final = json.loads((b / "final_choice.json").read_text())["final"]
    for style, path, score_column in (
        ("6b", b / "vol_h1/predictions_with_references.parquet", f"score_ind_{final}"),
        ("6d", d / "vol_h1/predictions.parquet", "score_pure_pure_all"),
    ):
        ref = pd.read_parquet(path, columns=cols + [score_column])
        rows = rows.merge(ref, on=keys, how="left", validate="one_to_one", suffixes=("", "_cnn"))
        if (not rows.label.eq(rows.label_cnn).all()
                or not pd.to_datetime(rows.decision_date).eq(pd.to_datetime(rows.decision_date_cnn)).all()
                or not rows[score_column].between(0, 1).all()):
            raise ValueError(f"{style} CNN keys, dates, labels or probabilities do not align")
        rows = rows.drop(columns=["label_cnn", "decision_date_cnn"]).rename(
            columns={score_column: f"{style}_cnn_probability"},
        )

    fm.register()
    store = Store(prepared)
    train = pd.read_parquet(b / "windows_train.parquet")
    _, median = ic.cell_rows(store, cell, "train", train)
    valid, _ = ic.cell_rows(store, cell, "validation", rows, median)
    # cell_rows sorts by position in the supplied windows.
    if not valid[keys + ["label"]].equals(rows[keys + ["label"]].astype(
            valid[keys + ["label"]].dtypes.to_dict())):
        raise ValueError("Shared rows or labels do not match the reconstructed v4 volatility task")
    pre = json.loads((b / "preprocessing.json").read_text())
    scales = json.loads((d / "strip_scales.json").read_text())
    ewsd, halflife = pre["ewsd_log_scale"], pre["halflife"]
    scaler = fm.scaler_from_state(pre["tab"])
    rv = regime.rv_tables(store, min_bars=settings["min_bars"])
    _, numbers = fm.numbers(store, rows, rv)
    log_rv = pi.log_rv(numbers)
    numbers = pd.DataFrame(scaler(numbers), columns=scaler.columns)
    charts = {
        "6b": ic.Charts(prepared, rows, final, ewsd, halflife),
        "6d": pi.PureCharts(prepared, rows, "pure_all", scales, halflife,
                            store.market_close, log_rv, numbers.to_numpy()),
    }
    rows["bar_of_day"] = rows.end_index % ic.LOOKBACK
    rows["abs_return"] = valid.forward_return.abs()
    rows["train_abs_return_median"] = rows.security_index.map(median)
    prep = {"ewsd": ewsd, "strips": scales, "tab": pre["tab"]}
    return rows, charts, numbers, prep


def answer_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in ("A", "B")]
    if any(len(tokens) != 1 for tokens in ids) or ids[0] == ids[1]:
        raise ValueError(f"Expected distinct single-token answers, got {ids}")
    return [tokens[0] for tokens in ids]


def model_inputs(processor, image, prompt, device):
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=[image], return_tensors="pt").to(device)


def make_prompt(style, features):
    prompt = LEGEND[style]
    if style == "6b":
        values = ", ".join(
            f"{name}={value:.8g}" if np.isfinite(value) else f"{name}=missing"
            for name, value in features.items()
        )
        prompt += "\nHistorical features: " + values
    return prompt + "\n" + QUESTION


def evaluation_prompt(style, features, shared=False):
    prompt = make_prompt(style, features)
    if shared and style == "6b":
        prompt = prompt.replace("Historical features:", "Standardized historical features:")
    return prompt


@torch.inference_mode()
def score(model, processor, image, prompt, ids):
    batch = model_inputs(processor, image, prompt, model.device)
    logits = model(**batch, use_cache=False).logits[0, -1].float()
    yes, no = logits[ids].tolist()
    probability = float(torch.sigmoid(torch.tensor(yes - no)))
    return yes, no, probability, batch["image_grid_thw"][0].tolist()


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run(prepared, bundle, output_root, n_rows=32, scale_rows=2000, seed=69, revision="main",
        shared_keys=None, references=None):
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Run with a GPU, e.g. Singularity exec --nv")
    if n_rows < 1 or (shared_keys is None and (n_rows > 247 or scale_rows < 2)):
        raise ValueError("Use 1–247 validation days and at least two training scale windows")
    if (shared_keys is None) != (references is None):
        raise ValueError("Supply both shared validation keys and CNN references")
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
    if shared_keys is None:
        rows, charts, numbers, prep = prepare_inputs(prepared, n_rows, scale_rows, seed)
    else:
        rows, charts, numbers, prep = prepare_shared_validation(
            prepared, shared_keys, references, n_rows, seed,
        )
        print(f"Aligned {len(rows):,} validation rows with 6b and 6d CNN predictions", flush=True)
    settings = {
        "mode": "zero_shot", "model": MODEL_ID, "requested_revision": revision,
        "model_revision": getattr(model.config, "_commit_hash", None),
        "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
        "bundle": bundle.name, "bundle_sha256": sha256(bundle),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "transformers": transformers.__version__, "seed": seed,
        "dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
        "requested_rows": n_rows, "validation_rows": len(rows),
        "scale_rows": scale_rows if shared_keys is None else None,
        "shared_validation_keys": str(shared_keys.resolve()) if shared_keys is not None else None,
        "shared_validation_keys_sha256": sha256(shared_keys) if shared_keys is not None else None,
        "cnn_references": str(references.resolve()) if references is not None else None,
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
    columns = ["security_index", "end_index", "decision_date", "bar_of_day",
               "label", "abs_return", "train_abs_return_median"]
    if shared_keys is not None:
        columns += ["6b_cnn_probability", "6d_cnn_probability"]
    pred = rows[columns].copy()
    summaries = []
    for style, chart in charts.items():
        start = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        scores = []
        for i in range(len(rows)):
            prompt = evaluation_prompt(style, numbers.iloc[i], shared=shared_keys is not None)
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
            "input_style": style, "mode": "zero_shot", "rows": len(y),
            "auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
            "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            "seconds": time.perf_counter() - start,
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        })
        if shared_keys is not None:
            p = pred[f"{style}_cnn_probability"]
            summaries.append({
                "input_style": style, "mode": "cnn", "rows": len(y),
                "auc": roc_auc_score(y, p) if y.nunique() == 2 else np.nan,
                "log_loss": log_loss(y, p, labels=[0, 1]), "brier": brier_score_loss(y, p),
            })
    table = pd.DataFrame(summaries)
    table.to_csv(output / "summary.csv", index=False)
    (output / "takeaways.md").write_text(
        "# Qwen zero-shot pilot\n\n"
        "Both input styles share the same validation keys.\n"
        + ("The CNN rows use saved scores on exactly these keys; the input scales "
           "come from the original CNN runs. No held-out testing was performed.\n"
           if shared_keys is not None else
           "Scales were fitted on sampled training windows. CNN comparison and "
           "held-out testing are separate runs.\n")
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
    reference = parser.add_mutually_exclusive_group()
    reference.add_argument("--probe-reference", type=Path, help="Completed zero-shot run directory")
    reference.add_argument("--lora-reference", type=Path, help="Completed frozen-probe run directory")
    reference.add_argument("--shared-validation", type=Path, help="Shared validation keys Parquet")
    parser.add_argument("--cnn-references", type=Path, help="Directory containing the original 6b and 6d runs")
    parser.add_argument("--train-rows", type=int, help="Optional probe training cap (pilot default: 2000)")
    parser.add_argument("--shared-train", type=Path, help="Canonical training keys for shared probe")
    parser.add_argument("--run-id", help="Stable directory name for restarting probe or LoRA")
    parser.add_argument("--lora-style", choices=("6b", "6d"), default="6d")
    parser.add_argument("--lora-epochs", type=int, default=3)
    parser.add_argument("--lora-max-train", type=int, help="Limit training rows for a smoke test")
    args = parser.parse_args()
    if args.lora_reference:
        from qwen_lora import run_lora
        run_lora(args.prepared, args.bundle, args.output, args.lora_reference,
                 args.lora_style, args.lora_epochs, args.lora_max_train, run_id=args.run_id)
    elif args.probe_reference:
        from qwen_probe import run_probe
        run_probe(args.prepared, args.bundle, args.output, args.probe_reference, args.train_rows,
                  shared_train=args.shared_train, run_id=args.run_id)
    else:
        run(args.prepared, args.bundle, args.output, args.rows, args.scale_rows, revision=args.revision,
            shared_keys=args.shared_validation, references=args.cnn_references)


if __name__ == "__main__":
    main()
