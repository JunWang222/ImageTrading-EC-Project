"""Train linear volatility probes on frozen Qwen prompt embeddings."""

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
from torch import nn

from qwen_eval import (
    LEGEND, MODEL_ID, QUESTION, Store, fm, ic, make_prompt, model_inputs, pi, regime, sha256,
)


KEYS = ["security_index", "end_index"]


def build_inputs(prepared, reference, settings, prep, train_rows):
    fm.register()
    store = Store(prepared)
    seed, stride = settings["seed"], settings["stride"]
    embargo = ic.embargo_dates(store, settings["embargo_sessions"])
    train = ic.decision_windows(store, "train", stride, embargo,
                                max_rows=settings["scale_rows"], seed=seed)
    valid = ic.decision_windows(store, "validation", stride, embargo, seed=seed)
    cell = {"target": "vol", "horizon": 1, "stride": stride}
    train, median = ic.cell_rows(store, cell, "train", train)
    valid, _ = ic.cell_rows(store, cell, "validation", valid, median)
    matched = reference[KEYS + ["label"]].merge(
        valid, on=KEYS, how="left", validate="one_to_one", suffixes=("_reference", ""),
    )
    if not matched.label.eq(matched.label_reference).all():
        raise ValueError("Reconstructed labels do not match the zero-shot validation keys")
    np.testing.assert_allclose(matched.security_index.map(median),
                               reference.train_abs_return_median)
    # The reference used scale_rows to fit image scales, not as a training-size limit.
    if len(train) < train_rows:
        train = ic.decision_windows(store, "train", stride, embargo,
                                    max_rows=2 * train_rows, seed=seed)
        train, _ = ic.cell_rows(store, cell, "train", train, median)
    train = train.sample(n=min(train_rows, len(train)), random_state=seed)
    if train.label.nunique() != 2 or matched.label.nunique() != 2:
        raise ValueError("Both splits must contain both labels")

    rv = regime.rv_tables(store, min_bars=settings["min_bars"])
    scaler = fm.scaler_from_state(prep["tab"])
    result = {}
    for split, rows in (("train", train), ("validation", matched)):
        _, numbers = fm.numbers(store, rows, rv)
        if numbers.columns.tolist() != settings["feature_order"]:
            raise ValueError("Feature order differs from the zero-shot run")
        result[split] = (rows, numbers, {
            "6b": ic.Charts(prepared, rows, "ewsd_tab_rv", prep["ewsd"], settings["halflife"]),
            "6d": pi.PureCharts(prepared, rows, "pure_all", prep["strips"], settings["halflife"],
                                 store.market_close, pi.log_rv(numbers), scaler(numbers)),
        })
    return result


@torch.no_grad()
def embed(model, processor, style, data, split, path):
    rows, numbers, charts = data
    vectors = []
    for i in range(len(rows)):
        image = Image.fromarray(charts[style].image(i)).convert("RGB")
        batch = model_inputs(processor, image, make_prompt(style, numbers.iloc[i]), model.device)
        hidden = model.model(**batch, use_cache=False, return_dict=True).last_hidden_state
        vectors.append(hidden[0, -1].float().cpu().clone())
        if (i + 1) % 100 == 0 or i + 1 == len(rows):
            print(f"{style} {split}: {i + 1}/{len(rows)} embeddings", flush=True)
    embeddings = torch.stack(vectors)
    torch.save({"embeddings": embeddings, "keys": rows[KEYS].to_numpy().tolist(),
                "labels": rows.label.tolist()}, path)
    return embeddings


def fit_probe(train, train_y, valid, valid_y, folder, seed, epochs, lr):
    torch.manual_seed(seed)
    mean = train.mean(0)
    scale = train.std(0, correction=0).clamp_min(1e-6)
    train, valid = (train - mean) / scale, (valid - mean) / scale
    head = nn.Linear(train.shape[1], 1)
    optim = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    train_eval = torch.randperm(len(train))[:512]
    history, best = [], float("inf")
    best_logits = None
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        head.train()
        total = 0.0
        for batch in torch.randperm(len(train)).split(64):
            optim.zero_grad()
            loss = loss_fn(head(train[batch]).flatten(), train_y[batch])
            loss.backward()
            optim.step()
            total += loss.item() * len(batch)
        head.eval()
        with torch.no_grad():
            logits = head(valid).flatten()
            val_loss = loss_fn(logits, valid_y).item()
            eval_loss = loss_fn(head(train[train_eval]).flatten(), train_y[train_eval]).item()
        history.append(dict(epoch=epoch, train_loss=total / len(train), train_eval_loss=eval_loss,
                            val_log_loss=val_loss, val_auc=roc_auc_score(valid_y, logits),
                            learning_rate=lr, seconds=time.perf_counter() - start))
        state = dict(model=head.state_dict(), optimizer=optim.state_dict(), epoch=epoch,
                     mean=mean, scale=scale, val_log_loss=val_loss, rng_state=torch.get_rng_state())
        torch.save(state, folder / "last.pt")
        if val_loss < best:
            best, best_logits = val_loss, logits.clone()
            torch.save(state, folder / "best.pt")
        pd.DataFrame(history).to_csv(folder / "history.csv", index=False)
    return best_logits, sum(p.numel() for p in head.parameters()), best


def run_probe(prepared, bundle, output_root, reference, train_rows=2000, epochs=12, lr=1e-3):
    import transformers
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if train_rows < 2 or epochs < 1 or lr <= 0 or not torch.cuda.is_available():
        raise ValueError("Probe requires CUDA, at least two training rows, and positive epochs and LR")
    settings = json.loads((reference / "settings.json").read_text())
    prep = json.loads((reference / "preprocessing.json").read_text())
    pred = pd.read_parquet(reference / "predictions.parquet")
    if (settings["model"] != MODEL_ID or settings["target"] != "vol_h1"
            or settings["question"] != QUESTION or settings["legend"] != LEGEND):
        raise ValueError("Probe reference must use the same model, target and prompts")
    digest = sha256(bundle)
    if digest != settings["bundle_sha256"]:
        raise ValueError("The v4 archive does not match the zero-shot bundle hash")
    revision = settings["model_revision"]
    if not revision:
        raise ValueError("The zero-shot reference does not record the exact Qwen revision")
    for style in LEGEND:
        if not np.isfinite(pred[f"{style}_prob_yes"]).all():
            raise ValueError(f"Incomplete zero-shot predictions for {style}")

    output = output_root / datetime.now(timezone.utc).strftime("qwen_probe_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
    if (type(processor.image_processor).__name__ != settings["image_processor_class"]
            or transformers.__version__ != settings["transformers"]):
        raise ValueError("Use the same Transformers version and processor as the zero-shot run")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval().requires_grad_(False)
    data = build_inputs(prepared, pred, settings, prep, train_rows)
    train, valid = (data[split][0] for split in ("train", "validation"))
    repo = Path(__file__).resolve().parents[2]
    (output / "settings.json").write_text(json.dumps({
        **settings, "mode": "probe", "zero_shot_reference": str(reference.resolve()),
        "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "train_rows_requested": train_rows, "train_rows": len(train), "epochs": epochs,
        "learning_rate": lr, "probe_batch_size": 64, "train_eval_rows": min(512, len(train)),
        "optimizer": "AdamW", "weight_decay": 0.01, "embedding": "last prompt token",
    }, indent=2) + "\n")
    (output / "preprocessing.json").write_text(json.dumps(prep, indent=2) + "\n")
    print(f"Saving probe results to {output}", flush=True)

    count = sum(p.numel() for p in model.parameters())
    summaries, parameters = [], []
    train_y = torch.tensor(train.label.tolist(), dtype=torch.float32)
    valid_y = torch.tensor(valid.label.tolist(), dtype=torch.float32)
    for style in LEGEND:
        folder = output / style
        folder.mkdir()
        vectors = [embed(model, processor, style, data[split], split, folder / f"{split}_embeddings.pt")
                   for split in ("train", "validation")]
        logits, n_head, val_loss = fit_probe(
            vectors[0], train_y, vectors[1], valid_y, folder, settings["seed"], epochs, lr,
        )
        p = logits.sigmoid().numpy()
        zero = pred[f"{style}_prob_yes"]
        pred[f"{style}_probe_logit"] = logits.numpy()
        pred[f"{style}_probe_probability"] = p
        summaries.append(dict(
            input_style=style, validation_rows=len(valid), auc=roc_auc_score(valid_y, logits),
            log_loss=val_loss, brier=brier_score_loss(valid_y, p),
            zero_shot_auc=roc_auc_score(valid_y, zero),
            zero_shot_log_loss=log_loss(valid_y, zero), zero_shot_brier=brier_score_loss(valid_y, zero),
        ))
        parameters.append(dict(input_style=style, frozen_qwen=count, total=count + n_head, trainable=n_head))
    pred.to_parquet(output / "predictions.parquet", index=False)
    pd.DataFrame(parameters).to_csv(output / "network_parameters.csv", index=False)
    table = pd.DataFrame(summaries)
    table.to_csv(output / "summary.csv", index=False)
    (output / "takeaways.md").write_text(
        "# Frozen Qwen probe pilot\n\n"
        "Linear heads were selected on validation log loss. Both input styles reuse "
        "the zero-shot validation keys, labels and image scales. See summary.csv for "
        "paired-set metrics and each style's history.csv for the learning curves.\n"
    )
    print(table.to_string(index=False))
    print("Saved:", output)
    return table
