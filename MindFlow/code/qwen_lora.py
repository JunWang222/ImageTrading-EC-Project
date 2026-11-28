"""Fine-tune Qwen attention adapters on the frozen probe's volatility windows."""

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
from torch.nn import functional as F

from qwen_eval import LEGEND, MODEL_ID, QUESTION, answer_ids, make_prompt, model_inputs, sha256
from qwen_probe import KEYS, build_inputs


def example_logit(model, processor, style, data, index, ids):
    _, numbers, charts = data
    image = Image.fromarray(charts[style].image(index)).convert("RGB")
    batch = model_inputs(processor, image, make_prompt(style, numbers.iloc[index]), model.device)
    scores = model(**batch, use_cache=False, logits_to_keep=1).logits[0, -1, ids].float()
    return scores[0] - scores[1]


@torch.no_grad()
def evaluate(model, processor, style, data, indices, ids):
    model.eval()
    logits = torch.stack([example_logit(model, processor, style, data, i, ids) for i in indices])
    labels = torch.tensor(data[0].label.iloc[indices].tolist(), device=logits.device, dtype=torch.float32)
    loss = F.binary_cross_entropy_with_logits(logits, labels).item()
    return loss, logits.cpu()


def verify_rows(data, probe, style):
    for split, (rows, _, _) in data.items():
        cache = torch.load(probe / style / f"{split}_embeddings.pt", map_location="cpu", weights_only=True)
        if rows[KEYS].to_numpy().tolist() != cache["keys"] or rows.label.tolist() != cache["labels"]:
            raise ValueError(f"{style} {split} keys or labels differ from the frozen probe")


def run_lora(prepared, bundle, output_root, probe, style="6d", epochs=3, max_train=None):
    import peft
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    if not torch.cuda.is_available() or epochs < 1 or (max_train is not None and max_train < 2):
        raise ValueError("Use CUDA, positive epochs and at least two training rows")
    settings = json.loads((probe / "settings.json").read_text())
    prep = json.loads((probe / "preprocessing.json").read_text())
    pred = pd.read_parquet(probe / "predictions.parquet")
    if (settings["model"] != MODEL_ID or settings["target"] != "vol_h1"
            or settings["question"] != QUESTION or settings["legend"] != LEGEND
            or sha256(bundle) != settings["bundle_sha256"]):
        raise ValueError("Model, prompts, target or data differ from the probe reference")
    revision = settings["model_revision"]
    if not revision:
        raise ValueError("Probe reference must record the exact Qwen revision")
    seed = settings["seed"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    print("Loading Qwen and verifying the probe's sample keys", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, revision=revision, use_fast=True)
    if (type(processor.image_processor).__name__ != settings["image_processor_class"]
            or transformers.__version__ != settings["transformers"]):
        raise ValueError("Use the probe's Transformers version and image processor")
    ids = answer_ids(processor.tokenizer)
    if ids != settings["answer_token_ids"]:
        raise ValueError("Tokenizer answer IDs differ from the reference")
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, revision=revision, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda")
    # A full regex excludes visual attention and limits updates to language attention.
    targets = r"model\.language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)"
    model = get_peft_model(base, LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.05, target_modules=targets, bias="none",
    ))
    data = build_inputs(prepared, pred, settings, prep, settings["train_rows_requested"])
    verify_rows(data, probe, style)
    train, valid = data["train"][0], data["validation"][0]
    train_indices = np.arange(min(max_train or len(train), len(train)))
    fixed_train = rng.choice(train_indices, size=min(128, len(train_indices)), replace=False)
    validation = np.arange(len(valid))
    labels = valid.label.to_numpy()

    output = output_root / datetime.now(timezone.utc).strftime("qwen_lora_%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[2]
    lr, accumulation, decay = 2e-5, 16, 0.01
    (output / "settings.json").write_text(json.dumps({
        **settings, "mode": "lora", "input_style": style, "probe_reference": str(probe.resolve()),
        "code_commit": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain"], text=True).strip()),
        "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "peft": peft.__version__,
        "epochs": epochs, "train_rows": len(train_indices), "train_rows_requested": max_train,
        "learning_rate": lr, "batch_size": 1, "gradient_accumulation": accumulation,
        "train_eval_rows": len(fixed_train), "optimizer": "AdamW", "weight_decay": decay,
        "lora_rank": 8, "lora_alpha": 16, "lora_dropout": 0.05, "lora_targets": targets,
        "embedding": None,
    }, indent=2) + "\n")
    (output / "preprocessing.json").write_text(json.dumps(prep, indent=2) + "\n")
    train.iloc[train_indices][KEYS + ["label"]].to_parquet(output / "train_keys.parquet", index=False)
    train.iloc[fixed_train][KEYS + ["label"]].to_parquet(output / "train_eval_keys.parquet", index=False)
    total = sum(p.numel() for p in model.parameters())
    parameters = [p for p in model.parameters() if p.requires_grad]
    pd.DataFrame([dict(input_style=style, total=total, trainable=sum(p.numel() for p in parameters))]).to_csv(
        output / "network_parameters.csv", index=False,
    )
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=decay)
    train_y = torch.tensor(train.label.tolist(), device=model.device, dtype=torch.float32)
    history, best = [], float("inf")
    print(f"Saving {style} LoRA results to {output}", flush=True)
    for epoch in range(1, epochs + 1):
        start = time.perf_counter()
        model.train()
        cumulative = 0.0
        order = rng.permutation(train_indices)
        for offset in range(0, len(order), accumulation):
            group = order[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for i in group:
                logit = example_logit(model, processor, style, data["train"], i, ids)
                loss = F.binary_cross_entropy_with_logits(logit, train_y[i])
                (loss / len(group)).backward()
                cumulative += loss.detach().item()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            seen = offset + len(group)
            if offset % (8 * accumulation) == 0 or seen == len(order):
                print(f"Epoch {epoch}: {seen}/{len(order)} train rows, loss={cumulative / seen:.4f}", flush=True)
        train_eval, _ = evaluate(model, processor, style, data["train"], fixed_train, ids)
        val_loss, logits = evaluate(model, processor, style, data["validation"], validation, ids)
        if not torch.isfinite(logits).all() or not np.isfinite(train_eval):
            raise RuntimeError(f"Non-finite loss or logits at epoch {epoch}")
        auc = roc_auc_score(labels, logits)
        history.append(dict(epoch=epoch, train_loss=cumulative / len(order), train_eval_loss=train_eval,
                            val_log_loss=val_loss, val_auc=auc, learning_rate=lr,
                            seconds=time.perf_counter() - start))
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        model.save_pretrained(output / "last_adapter")
        torch.save(dict(epoch=epoch, optimizer=optimizer.state_dict(), rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()),
                   output / "last.pt")
        if val_loss < best:
            best = val_loss
            model.save_pretrained(output / "best_adapter")
            probability = logits.sigmoid().numpy()
            pred[f"{style}_lora_logit"] = logits.numpy()
            pred[f"{style}_lora_probability"] = probability
            pred.to_parquet(output / "predictions.parquet", index=False)
            result = []
            for method, scores in (
                ("zero_shot", pred[f"{style}_prob_yes"]),
                ("probe", pred[f"{style}_probe_probability"]),
                ("lora", probability),
            ):
                result.append(dict(input_style=style, mode=method, rows=len(pred),
                                   auc=roc_auc_score(labels, scores),
                                   log_loss=log_loss(labels, scores, labels=[0, 1]),
                                   brier=brier_score_loss(labels, scores),
                                   best_epoch=epoch if method == "lora" else None))
            table = pd.DataFrame(result)
            table.to_csv(output / "summary.csv", index=False)
        print(f"Epoch {epoch}: train eval loss={train_eval:.4f}, val loss={val_loss:.4f}, val AUC={auc:.4f}")
    (output / "takeaways.md").write_text(
        "# Qwen LoRA validation pilot\n\n"
        "LoRA adapters were selected on validation log loss. Summary metrics use the "
        "same validation windows for zero-shot, probe and LoRA. No held-out test was run.\n"
    )
    print(table.to_string(index=False))
    print("Saved:", output)
    return table
