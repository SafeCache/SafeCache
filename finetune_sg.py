import argparse
import csv
import gc
import json
import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

DEFAULT_MODEL = "./models/ShieldGemma-9B"

DEFAULT_GUIDELINE = (
    '"Harmful Content": The prompt shall not contain or seek generation of '
    "content that: facilitates violent crimes (terrorism, murder, assault); "
    "facilitates non-violent crimes (fraud, theft, hacking, drug trafficking); "
    "facilitates sex crimes or non-consensual acts; sexually exploits minors "
    "(CSAM); constitutes defamation or false factual claims; provides "
    "specialized dangerous advice (medical, legal, financial) likely to cause "
    "harm; violates personal privacy; infringes intellectual property; provides "
    "instructions for weapons of mass destruction; promotes hate or "
    "discrimination based on protected characteristics; encourages self-harm or "
    "suicide; contains explicit sexual content; attempts to interfere with "
    "elections; or abuses code interpreter / tool-call capabilities."
)

SG_LABEL = {"unsafe": "Yes", "safe": "No"}

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

LEARNING_RATE = 2e-4
BATCH_SIZE    = 1
GRAD_ACCUM    = 16
MAX_SEQ_LEN   = 1024


def _wg_label(d):
    if d.get("prompt_harm_label") is None:
        return None
    return "unsafe" if d["prompt_harm_label"] == "harmful" else "safe"

def _bt_label(d):
    return "safe" if d.get("is_safe") else "unsafe"

def _lmsys_label(d):
    return d.get("label")

def _tc_label(d):
    return "unsafe" if int(d.get("toxicity", 0)) == 1 else "safe"

def _aegis2_label(d):
    return d.get("prompt_label")

DATASET_REGISTRY = {
    "wildguardmix": {
        "train":    "data/wildguardmix/train.json",
        "test":     "data/wildguardmix/test.json",
        "label_fn": _wg_label,
        "prompt_field": "prompt",
    },
    "beavertails": {
        "train":    "data/beavertails/beavertails_train.jsonl",
        "test":     "data/beavertails/beavertails_test.jsonl",
        "label_fn": _bt_label,
        "prompt_field": "prompt",
    },
    "lmsys": {
        "train":    "data/lmsys-chat-1m/train.json",
        "test":     "data/lmsys-chat-1m/test.json",
        "label_fn": _lmsys_label,
        "prompt_field": "prompt",
    },
    "toxicchat": {
        "train":    "data/toxic-chat/toxic-chat_train.csv",
        "test":     "data/toxic-chat/toxic-chat_test.csv",
        "label_fn": _tc_label,
        "prompt_field": "user_input",
    },
    "aegis2": {
        "train":    "data/aegis2/aegis2_train.csv",
        "test":     "data/aegis2/aegis2_test.csv",
        "label_fn": _aegis2_label,
        "prompt_field": "prompt",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",   type=str, nargs="+", required=True,
                   choices=list(DATASET_REGISTRY.keys()),
                   help="One or more datasets to fine-tune on (combined)")
    p.add_argument("--model",     type=str, default=DEFAULT_MODEL,
                   help="Path to base guard model (default: ShieldGemma-9B)")
    p.add_argument("--guideline", type=str, default=DEFAULT_GUIDELINE,
                   help="Safety policy guideline string passed to ShieldGemma")
    p.add_argument("--limit",     type=int, default=None,
                   help="Cap training samples (for smoke tests)")
    p.add_argument("--eval-only", action="store_true",
                   help="Skip training, load saved adapter and evaluate")
    p.add_argument("--epochs",    type=int, default=4)
    p.add_argument("--lr",        type=float, default=LEARNING_RATE,
                   help=f"Learning rate (default: {LEARNING_RATE})")
    p.add_argument("--no-merge",  action="store_true",
                   help="Save adapter only (don't merge weights at eval time)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Override output directory (default: models/{model_name}-{dataset}-ft)")
    return p.parse_args()


def load_data(path, label_fn, limit=None, prompt_field="prompt"):
    samples = []
    is_csv  = path.endswith(".csv")
    with open(path, "r", newline="" if is_csv else None) as f:
        rows = csv.DictReader(f) if is_csv else (json.loads(l) for l in f)
        for d in rows:
            label = label_fn(d)
            if label is None:
                continue
            prompt = d.get(prompt_field, "").strip()
            if not prompt:
                continue
            samples.append({"prompt": prompt, "label": label})
            if limit and len(samples) >= limit:
                break
    return samples


def dedup(samples):
    seen, out = set(), []
    for s in samples:
        if s["prompt"] not in seen:
            seen.add(s["prompt"])
            out.append(s)
    removed = len(samples) - len(out)
    if removed:
        print(f"  Dedup: removed {removed:,} duplicates ({len(out):,} unique remain)")
    return out


class GuardDataset(Dataset):
    def __init__(self, samples, tokenizer, guideline, max_len):
        self.items = []
        skipped = 0
        for s in samples:
            messages  = [{"role": "user", "content": s["prompt"]}]
            input_ids = tokenizer.apply_chat_template(
                messages, guideline=guideline,
                add_generation_prompt=True, tokenize=True,
            )
            if not isinstance(input_ids, list):
                input_ids = input_ids["input_ids"]
            if not isinstance(input_ids, list):
                input_ids = input_ids.squeeze().tolist()

            sg_label  = SG_LABEL[s["label"]]
            label_ids = tokenizer.encode(sg_label + "\n", add_special_tokens=False)
            label_ids += [tokenizer.eos_token_id]

            full_ids = input_ids + label_ids
            labels   = [-100] * len(input_ids) + label_ids

            if len(full_ids) > max_len:
                keep = max_len - len(label_ids)
                if keep < 20:
                    skipped += 1
                    continue
                full_ids = input_ids[:keep] + label_ids
                labels   = [-100] * keep + label_ids

            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels":    torch.tensor(labels,   dtype=torch.long),
            })

        if skipped:
            print(f"  Skipped {skipped} samples exceeding max_seq_len={max_len}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    max_len       = max(b["input_ids"].shape[0] for b in batch)
    input_ids_out = torch.zeros(len(batch), max_len, dtype=torch.long)
    labels_out    = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_out      = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].shape[0]
        input_ids_out[i, :n] = b["input_ids"]
        labels_out[i, :n]    = b["labels"]
        attn_out[i, :n]      = 1
    return {"input_ids": input_ids_out, "labels": labels_out, "attention_mask": attn_out}


def unload(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()


YES_TOKEN_ID = None
NO_TOKEN_ID  = None

def init_classification_tokens(tokenizer):
    global YES_TOKEN_ID, NO_TOKEN_ID
    YES_TOKEN_ID = tokenizer.convert_tokens_to_ids("Yes")
    NO_TOKEN_ID  = tokenizer.convert_tokens_to_ids("No")


def _run_guard(model, tokenizer, prompt, guideline):
    messages = [{"role": "user", "content": prompt}]
    inputs   = tokenizer.apply_chat_template(
        messages, guideline=guideline,
        add_generation_prompt=True, return_tensors="pt",
    )
    if isinstance(inputs, torch.Tensor):
        inputs = {"input_ids": inputs}
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    yes_no = logits[0, -1, [YES_TOKEN_ID, NO_TOKEN_ID]]
    return 0 if yes_no[1] > yes_no[0] else 1


def print_metrics(title, labels, preds):
    acc  = accuracy_score(labels, preds)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    n    = len(labels)
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'-'*60}")
    print(f"  Samples   : {n:,}  "
          f"(harmful={sum(labels):,}  harmless={n-sum(labels):,})")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f}  (positive = harmful)")
    print(f"  Recall    : {rec:.4f}  (positive = harmful)")
    print(f"  F1        : {f1:.4f}")
    print("=" * 60)
    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1}


def step1_prepare(samples, tokenizer, guideline):
    print(f"\n{'='*60}")
    print(f" Step 1 — Data Preparation  ({len(samples):,} samples)")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    ds = GuardDataset(samples, tokenizer, guideline, MAX_SEQ_LEN)
    elapsed  = time.perf_counter() - t0
    n_safe   = sum(1 for s in samples if s["label"] == "safe")
    n_unsafe = sum(1 for s in samples if s["label"] == "unsafe")
    print(f"  Tokenized {len(ds):,} / {len(samples):,} samples  "
          f"(safe={n_safe:,}  unsafe={n_unsafe:,})")
    print(f"  Step 1 time: {elapsed:.1f}s")
    return ds, elapsed


def step2_finetune(model, tokenizer, ds, epochs, grad_accum, lr=LEARNING_RATE):
    print(f"\n{'='*60}")
    print(f" Step 2 — Fine-tuning  "
          f"(LoRA r={LORA_R}, lr={lr}, {epochs} epoch(s), {len(ds):,} samples)")
    print(f"{'='*60}")

    loader      = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    total_steps = max(1, len(loader) * epochs // grad_accum)
    optimizer   = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )

    model.train()
    model.enable_input_require_grads()
    device = next(model.parameters()).device

    t0 = time.perf_counter()
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=70)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            outputs    = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
            loss       = outputs.loss / grad_accum
            loss.backward()
            epoch_loss += outputs.loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        print(f"  Epoch {epoch+1} avg loss: {epoch_loss/len(loader):.4f}")

    elapsed = time.perf_counter() - t0
    print(f"  Step 2 time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    return elapsed


def step3_evaluate(model, tokenizer, test_path, label_fn, guideline,
                   limit=None, prompt_field="prompt", dataset_name=""):
    print(f"\n{'='*60}")
    print(f" Step 3 — Evaluation on test set (logit-based){f'  [{dataset_name}]' if dataset_name else ''}")
    print(f"{'='*60}")
    test_limit   = (limit // 5) if limit else None
    test_samples = load_data(test_path, label_fn, limit=test_limit, prompt_field=prompt_field)
    n            = len(test_samples)
    print(f"  {n:,} test samples")

    model.eval()
    t0 = time.perf_counter()
    preds, labels = [], []
    for s in tqdm(test_samples, desc="eval", ncols=70):
        pred = _run_guard(model, tokenizer, s["prompt"], guideline)
        preds.append(pred)
        labels.append(0 if s["label"] == "safe" else 1)

    elapsed = time.perf_counter() - t0
    print(f"  Step 3 time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print(f"  Avg per sample: {elapsed/n*1000:.0f} ms")
    m = print_metrics(f"Fine-tuned SG — {dataset_name or 'Test'} Evaluation", labels, preds)
    print(f"[RESULT] dataset={dataset_name or 'test'} f1={m['f1']:.4f} prec={m['prec']:.4f} rec={m['rec']:.4f} acc={m['acc']:.4f}")
    return elapsed


def main():
    args        = parse_args()
    datasets    = args.dataset
    dataset_tag = "+".join(datasets)
    model_name  = os.path.basename(args.model.rstrip("/"))
    output_dir  = args.output_dir or f"./models/{model_name}-{dataset_tag}-ep{args.epochs}-lr{args.lr}-ft"
    guideline   = args.guideline

    print("=" * 60)
    print(f" Guard model : {model_name}")
    print(f" Dataset     : {dataset_tag}  (train only, not cache)")
    print(f" LoRA r={LORA_R}, alpha={LORA_ALPHA}, lr={args.lr}, "
          f"eff_batch={GRAD_ACCUM}")
    print(f" Output      : {output_dir}")
    print("=" * 60)

    tok = AutoTokenizer.from_pretrained(args.model)
    init_classification_tokens(tok)
    print(f"  Yes token id={YES_TOKEN_ID}  No token id={NO_TOKEN_ID}")
    t1 = t2 = 0.0

    if not args.eval_only:
        train_samples = []
        for ds_name in datasets:
            cfg = DATASET_REGISTRY[ds_name]
            print(f"\nLoading training data from {cfg['train']} ...")
            samples = load_data(cfg["train"], cfg["label_fn"], limit=args.limit,
                                prompt_field=cfg.get("prompt_field", "prompt"))
            samples = dedup(samples)
            n_safe   = sum(1 for s in samples if s["label"] == "safe")
            n_unsafe = sum(1 for s in samples if s["label"] == "unsafe")
            print(f"  {len(samples):,} samples  (safe={n_safe:,}  unsafe={n_unsafe:,})")
            train_samples.extend(samples)

        if len(datasets) > 1:
            train_samples = dedup(train_samples)
            n_safe   = sum(1 for s in train_samples if s["label"] == "safe")
            n_unsafe = sum(1 for s in train_samples if s["label"] == "unsafe")
            print(f"\n  Combined (after dedup): {len(train_samples):,}  "
                  f"(safe={n_safe:,}  unsafe={n_unsafe:,})")

        ds, t1 = step1_prepare(train_samples, tok, guideline)

        print(f"\nLoading {args.model} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        t2 = step2_finetune(model, tok, ds, args.epochs, GRAD_ACCUM, lr=args.lr)

        print(f"\nSaving LoRA adapter to {output_dir} ...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tok.save_pretrained(output_dir)

        if not args.no_merge:
            print("  Merging LoRA weights into base model for faster inference ...")
            model = model.merge_and_unload()

    else:
        datasets = args.dataset
        print(f"\nLoading {args.model} + adapter from {output_dir} ...")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto",
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(model, output_dir)
        if not args.no_merge:
            print("  Merging LoRA weights ...")
            model = model.merge_and_unload()

    t3 = 0.0
    for ds_name in datasets:
        eval_cfg = DATASET_REGISTRY[ds_name]
        t3 += step3_evaluate(model, tok, eval_cfg["test"], eval_cfg["label_fn"],
                             guideline, limit=args.limit,
                             prompt_field=eval_cfg.get("prompt_field", "prompt"),
                             dataset_name=ds_name)

    total = t1 + t2 + t3
    print(f"\n{'='*60}")
    print(" Total Timing Summary")
    print(f"{'-'*60}")
    print(f"  Step 1 (data prep)  : {t1:.1f}s")
    print(f"  Step 2 (fine-tuning): {t2:.1f}s  ({t2/60:.1f} min)")
    print(f"  Step 3 (evaluation) : {t3:.1f}s  ({t3/60:.1f} min)")
    print(f"  Total               : {total:.1f}s  ({total/60:.1f} min)")
    print("=" * 60)


if __name__ == "__main__":
    main()
