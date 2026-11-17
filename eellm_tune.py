import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import semantic_cache_eval as sce
import early_exit_eval as eee
import eellm_exit as ee


class _StopForward(Exception):
    pass


VERDICT_WORDS = {
    "causal":      {0: "safe", 1: "unsafe"},
    "lg4":         {0: "safe", 1: "unsafe"},
    "shieldgemma": {0: "No",   1: "Yes"},
}


def verdict_ids(tok, arch):
    base = tok.tokenizer if arch == "lg4" else tok
    return {lbl: base.encode(w, add_special_tokens=False)
            for lbl, w in VERDICT_WORDS[arch].items()}


def build_sequences(prompts, labels, tok, arch, newline_id, max_seq_len):
    vids = verdict_ids(tok, arch)
    seqs, vlens, n_skip = [], [], 0
    for p, lbl in zip(tqdm(prompts, desc="  tokenize", ncols=72), labels):
        inp = eee.build_exit_input(p, tok, arch, newline_id)
        ids = inp["input_ids"][0]
        v   = torch.tensor(vids[int(lbl)], dtype=ids.dtype)
        if ids.shape[0] + v.shape[0] > max_seq_len:
            n_skip += 1
            continue
        seqs.append(torch.cat([ids, v]))
        vlens.append(v.shape[0])
    return seqs, vlens, n_skip


def capture_hidden(model, layer_module, ids, device):
    buf = {}

    def _hook(mod, _inp, output):
        buf["h"] = output[0] if isinstance(output, tuple) else output
        raise _StopForward

    handle = layer_module.register_forward_hook(_hook)
    try:
        with torch.no_grad():
            model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
    except _StopForward:
        pass
    finally:
        handle.remove()
    return buf["h"][0]


def place_backbone(model, arch, exit_layer_idx, device, offload=True):
    layers = sce.get_layers(model, arch)
    for i, layer in enumerate(layers):
        layer.to(device if (i <= exit_layer_idx or not offload) else "cpu")
    if arch == "lg4":
        for name in ("vision_model", "multi_modal_projector"):
            m = getattr(model, name, None)
            if m is not None:
                m.to(device if not offload else "cpu")
    torch.cuda.empty_cache()
    if offload:
        n_off = len(layers) - exit_layer_idx - 1
        print(f"  [partial load] {n_off} decoder layers above exit "
              f"{exit_layer_idx} offloaded to CPU")


def _chunk_ce(head, x, t, n_tok):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = head.logits(x)
    return F.cross_entropy(logits.float(), t, reduction="sum") / n_tok


def seq_exit_loss(head, h, ids, vl, args, batch, device):
    from torch.utils.checkpoint import checkpoint
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = head.hidden(h[:-1])
    targets = ids[1:].to(device)
    if args.verdict_only:
        x, targets = x[-vl:], targets[-vl:]
    n_tok = targets.numel()
    loss = 0.0
    for s in range(0, n_tok, args.loss_chunk):
        loss = loss + checkpoint(_chunk_ce, head, x[s:s + args.loss_chunk],
                                 targets[s:s + args.loss_chunk], n_tok,
                                 use_reentrant=False)
    (loss / batch).backward()
    return float(loss.detach())


def exit_size_from_args(args):
    return {"mlp_blocks":  args.exit_mlp_blocks,
            "mlp_width":   args.exit_mlp_width,
            "head_hidden": args.exit_head_hidden}


def tune_one_exit(model, tok, arch, layer_idx, seqs, vlens, args, device):
    hp   = ee.EELLM_TUNE_HPARAMS
    place_backbone(model, arch, layer_idx, device, offload=not args.no_offload)
    if args.exit_lm:
        head = ee.build_lm_exit_head(model, arch, layer_idx, args.exit_lm,
                                     random_init=args.exit_lm_random_init)
    else:
        head = ee.build_exit_head(model, arch, layer_idx,
                                  size=exit_size_from_args(args))
    head = head.to(device=device, dtype=torch.float32)
    head.train()
    for p in head.parameters():
        p.requires_grad_(True)

    n = len(seqs)
    steps_per_epoch = math.ceil(n / hp["global_batch_size"])
    total_steps = args.train_iters or steps_per_epoch * args.epochs
    warmup = max(1, int(total_steps * hp["lr_warmup_fraction"]))
    opt = torch.optim.Adam(head.parameters(), lr=hp["lr"],
                           betas=(hp["adam_beta1"], hp["adam_beta2"]),
                           eps=hp["adam_eps"], fused=True)
    floor = hp["min_lr"] / hp["lr"]

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        frac = (step - warmup) / max(1, total_steps - warmup)
        return max(floor, 1.0 - (1.0 - floor) * frac)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    layer_module = sce.get_layers(model, arch)[layer_idx]

    n_params = sum(p.numel() for p in head.parameters())
    print(f"\n  [exit @ layer {layer_idx}] trainable params: {n_params / 1e6:.0f}M  "
          f"steps: {total_steps} (batch {hp['global_batch_size']}, warmup {warmup})  "
          f"~{n_params * 16 / 2**30:.1f} GiB fp32 weights+grads+Adam")

    rng  = np.random.default_rng(sce.SEED + layer_idx)
    ema  = None
    t0   = time.time()
    step = 0
    micro_iter = iter(())

    def epoch_iter():
        while True:
            order = rng.permutation(n)
            for j in order:
                yield j

    micro_iter = epoch_iter()
    pbar = tqdm(range(total_steps), desc=f"  tune L{layer_idx}", ncols=90)
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(hp["global_batch_size"]):
            j   = next(micro_iter)
            ids = seqs[j]
            h   = capture_hidden(model, layer_module, ids, device).float()
            if ee.is_seq_exit(head):
                loss_sum += seq_exit_loss(head, h, ids, vlens[j], args,
                                          hp["global_batch_size"], device)
                continue
            if args.verdict_only:
                vl      = vlens[j]
                h_in    = h[-vl - 1:-1]
                targets = ids[-vl:].to(device)
            else:
                h_in    = h[:-1]
                targets = ids[1:].to(device)
            n_tok = targets.numel()
            for s in range(0, n_tok, args.loss_chunk):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = head(h_in[s:s + args.loss_chunk])
                loss_c = F.cross_entropy(logits.float(),
                                         targets[s:s + args.loss_chunk],
                                         reduction="sum") / n_tok
                (loss_c / hp["global_batch_size"]).backward()
                loss_sum += float(loss_c.detach())
        torch.nn.utils.clip_grad_norm_(head.parameters(), hp["clip_grad"])
        opt.step()
        sched.step()
        avg = loss_sum / hp["global_batch_size"]
        ema = avg if ema is None else 0.98 * ema + 0.02 * avg
        if step % args.log_every == 0 or step == total_steps - 1:
            pbar.set_postfix(loss=f"{ema:.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")

    print(f"  [exit @ layer {layer_idx}] done in {(time.time() - t0) / 60:.1f} min  "
          f"final loss (ema) {ema:.4f}")
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--cache-dataset", nargs="+",
                    choices=list(sce.DATASET_PATHS.keys()),
                    default=["wildguardmix", "aegis2", "toxicchat"],
                    help="Datasets whose cache/index split (80%%) is the tuning "
                         "corpus — same split the semantic cache indexes")
    ap.add_argument("--exit-layers", type=int, nargs="+", default=None,
                    help="0-based layer indices whose output feeds an exit "
                         "(default: 1/4 and 1/2 depth, as EE-LLM)")
    ap.add_argument("--train-full", action="store_true",
                    help="Tune on each dataset's full train_full split instead "
                         "of the 80%% cache/index split (default output dir "
                         "gets a -trainfull suffix)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--train-iters", type=int, default=None,
                    help="Optimizer steps; overrides --epochs "
                         "(EE-Tuning paper value: 40000)")
    ap.add_argument("--verdict-only", action="store_true",
                    help="Restrict the LM loss to the appended verdict "
                         "token(s) instead of the paper's full-sequence loss")
    ap.add_argument("--loss-chunk", type=int, default=256,
                    help="Positions per logits/CE chunk (memory bound only — "
                         "the loss is mathematically unchanged)")
    ap.add_argument("--no-offload", action="store_true",
                    help="Keep the whole backbone on GPU instead of "
                         "offloading the decoder layers above the exit "
                         "(EE-Tuning partial loading)")
    ap.add_argument("--exit-mlp-blocks", type=int, default=1,
                    help="Capacity ablation: residual MLP blocks per exit "
                         "(block j copied from layer exit+j; default 1 = EE-LLM)")
    ap.add_argument("--exit-mlp-width", type=int, default=1,
                    help="Capacity ablation: exit MLP intermediate size x this "
                         "factor (function-preserving widening; default 1)")
    ap.add_argument("--exit-head-hidden", type=int, default=0,
                    help="Capacity ablation: hidden size of a non-linear "
                         "d->H->V branch added to the LM head (0 = none)")
    ap.add_argument("--exit-lm", type=str, default=None,
                    help="Capacity ablation: replace the trainable exit with this "
                         "pretrained general LM (e.g. ./models/Llama-3.2-1B; must "
                         "share the guard's vocabulary), fed the full-sequence "
                         "exit-layer hidden states through a linear adapter")
    ap.add_argument("--exit-lm-random-init", action="store_true",
                    help="With --exit-lm: same LM architecture, random weights "
                         "(control for the LM's pretraining)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap tuning prompts (smoke tests)")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default models/eellm-exits-<model-name>)")
    args = ap.parse_args()

    sce.set_seed()
    cfg = sce.resolve_model_cfg(args.model)
    exit_size = exit_size_from_args(args)
    if args.exit_lm and exit_size != ee.DEFAULT_EXIT_SIZE:
        raise SystemExit("--exit-lm replaces the MLP exit; drop the --exit-mlp-*/"
                         "--exit-head-hidden options")
    if args.exit_lm_random_init and not args.exit_lm:
        raise SystemExit("--exit-lm-random-init needs --exit-lm")
    size_tag = (f"-lm-{os.path.basename(args.exit_lm.rstrip('/'))}"
                + ("-rand" if args.exit_lm_random_init else "")
                if args.exit_lm else ee.exit_size_tag(exit_size))
    out_dir = args.out or (ee.heads_dir_for(args.model)
                           + ("-trainfull" if args.train_full else "")
                           + size_tag)
    if os.path.exists(os.path.join(out_dir, "config.json")):
        raise SystemExit(f"Refusing to overwrite existing exit heads at {out_dir}")
    hp = ee.EELLM_TUNE_HPARAMS

    print("=" * 74)
    print("  EE-Tuning (EE-LLM baseline) — tune early-exit layers, frozen backbone")
    print(f"  Model       : {args.model}  (arch={cfg['arch']})")
    split_name = "train_full" if args.train_full else "index split (80%)"
    print(f"  Tuning data : {split_name} of {args.cache_dataset}")
    print(f"  Optimizer   : Adam({hp['adam_beta1']}, {hp['adam_beta2']}, "
          f"eps={hp['adam_eps']})  lr {hp['lr']}→{hp['min_lr']} linear, "
          f"warmup {hp['lr_warmup_fraction']}, clip {hp['clip_grad']}")
    print(f"  Batch       : {hp['global_batch_size']} sequences/step, "
          f"seq len ≤ {hp['max_seq_len']}")
    if args.exit_lm:
        print(f"  Exit        : LM exit {args.exit_lm}"
              f"{' (random init)' if args.exit_lm_random_init else ' (pretrained)'}"
              f" on full-sequence exit-layer states")
    else:
        print(f"  Exit size   : {exit_size}"
              f"{'  (EE-LLM default)' if exit_size == ee.DEFAULT_EXIT_SIZE else ''}")
    print(f"  Output      : {out_dir}")
    print("=" * 74)

    prompts, labels = [], []
    for ds in args.cache_dataset:
        if args.train_full:
            path = sce.DATASET_PATHS[ds].get("train_full")
            if path is None:
                raise ValueError(f"Dataset '{ds}' has no train_full split")
            tp, tl = sce.load_data(path, ds)
            tp, tl, n_rm = sce.dedup(tp, tl)
            prompts += list(tp)
            labels  += list(tl)
            print(f"  [{ds}] train_full: {len(tp):,} "
                  f"(dedup removed {n_rm:,})")
        else:
            path = sce.DATASET_PATHS[ds]["cache_data"]
            if path is None:
                raise ValueError(f"Dataset '{ds}' has no cache split")
            tp, tl = sce.load_data(path, ds)
            tp, tl, n_rm = sce.dedup(tp, tl)
            idx_mask, _ = sce.stratified_split(tl)
            prompts += [p for p, m in zip(tp, idx_mask) if m]
            labels  += list(tl[idx_mask])
            print(f"  [{ds}] index split: {int(idx_mask.sum()):,} "
                  f"(dedup removed {n_rm:,})")
    labels = np.array(labels)
    if args.limit and len(prompts) > args.limit:
        rng = np.random.default_rng(sce.SEED)
        sel = np.sort(rng.choice(len(prompts), args.limit, replace=False))
        prompts = [prompts[i] for i in sel]
        labels  = labels[sel]
    print(f"  Tuning prompts: {len(prompts):,}  (harmful={int(labels.sum()):,}  "
          f"harmless={int((labels == 0).sum()):,})")

    print("\n  Loading model ...")
    model, tok = sce.load_model(cfg)
    arch     = cfg["arch"]
    device   = next(model.parameters()).device
    n_layers = len(sce.get_layers(model, arch))
    for p in model.parameters():
        p.requires_grad_(False)

    exit_layers = sorted(args.exit_layers or ee.default_exit_layers(n_layers))
    assert all(0 <= li < n_layers for li in exit_layers)
    print(f"  num_layers={n_layers}  exit layers (0-based): {exit_layers}  "
          f"(EE-LLM: 1/4 and 1/2 depth)")

    if args.exit_lm:
        from transformers import AutoTokenizer
        base_tok = tok.tokenizer if arch == "lg4" else tok
        lm_tok = AutoTokenizer.from_pretrained(args.exit_lm)
        if lm_tok.get_vocab() != base_tok.get_vocab():
            raise SystemExit(f"{args.exit_lm} tokenizer differs from the guard's")
        print(f"  LM exit tokenizer == guard tokenizer ({len(lm_tok):,} tokens)")

    newline_id = eee.resolve_newline_id(tok, arch, is_finetuned=cfg["ft"] is not None)
    seqs, vlens, n_skip = build_sequences(prompts, labels, tok, arch,
                                          newline_id, hp["max_seq_len"])
    print(f"  Sequences: {len(seqs):,}  (skipped {n_skip:,} over "
          f"{hp['max_seq_len']} tokens)")

    heads = {}
    for li in exit_layers:
        heads[li] = tune_one_exit(model, tok, arch, li, seqs, vlens, args, device)
        heads[li].to(dtype=torch.bfloat16)
        torch.cuda.empty_cache()

    meta = {
        "method":       "ee-llm / ee-tuning",
        "base_model":   args.model,
        "arch":         arch,
        "n_layers":     n_layers,
        "exit_layers":  exit_layers,
        "exit_arch":    (f"lm exit: RMSNorm+linear adapter -> {args.exit_lm} "
                         f"({'random init' if args.exit_lm_random_init else 'pretrained'}),"
                         " full-sequence input" if args.exit_lm else
                         "mlp+norm, untied copy-init output embedding"
                         + ("" if exit_size == ee.DEFAULT_EXIT_SIZE else
                            f" (enlarged: {ee.exit_size_tag(exit_size)[1:]})")),
        "exit_size":    exit_size,
        "exit_type":    "lm" if args.exit_lm else "mlp",
        "exit_lm":      args.exit_lm,
        "exit_lm_random_init": bool(args.exit_lm_random_init),
        "loss":         "verdict_only" if args.verdict_only else "full_sequence_lm",
        "tuning_data":  args.cache_dataset,
        "tuning_split": "train_full" if args.train_full else "cache_index_80",
        "n_sequences":  len(seqs),
        "epochs":       args.epochs,
        "train_iters":  args.train_iters,
        "hparams":      ee.EELLM_TUNE_HPARAMS,
    }
    ee.save_exit_heads(heads, meta, out_dir)
    sce.unload(model)


if __name__ == "__main__":
    main()
