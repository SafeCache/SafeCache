import argparse, csv, itertools, math, os, time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import semantic_cache_eval as sce
from semantic_cache_eval import _EarlyExit


LTT_EPSILON    = 0.05
LAMBDA_GRID    = np.round(np.arange(1.0, -1e-9, -0.05), 4)
META_MAX_TRAIN = 20000
CONF_METHODS   = ("softmax_diff", "softmax_max", "state", "meta", "verdict_margin")


def get_verdict_head(model, arch):
    if arch == "lg4":
        lm    = model.language_model
        inner = lm.model if hasattr(lm, "model") else lm
        norm  = inner.norm
        head  = getattr(lm, "lm_head", None) or model.lm_head
    else:
        norm = model.model.norm
        head = model.lm_head
    cfg_t   = getattr(model.config, "text_config", model.config)
    softcap = getattr(cfg_t, "final_logit_softcapping", None)

    def verdict_logits(h):
        with torch.no_grad():
            x      = norm(h.to(norm.weight.dtype))
            logits = head(x)
            if softcap:
                logits = torch.tanh(logits / softcap) * softcap
            return logits.float()
    return verdict_logits


def resolve_newline_id(tok, arch, is_finetuned):
    if arch in ("causal", "lg4") and not is_finetuned:
        return sce.get_newline_id(tok, arch)
    return None


def build_exit_input(prompt, tok, arch, newline_id):
    inp = sce.make_input(prompt, tok, arch)
    if newline_id is not None:
        ids = inp["input_ids"]
        nl  = torch.tensor([[newline_id]], dtype=ids.dtype)
        inp["input_ids"] = torch.cat([ids, nl], dim=1)
        if "attention_mask" in inp:
            inp["attention_mask"] = torch.cat(
                [inp["attention_mask"], torch.ones_like(nl)], dim=1)
    return inp


def guard_call(model, inp, tok, arch, logit_mode):
    if logit_mode:
        return model(**inp, logits_to_keep=1)
    return model.generate(**inp, max_new_tokens=20,
                          pad_token_id=sce.get_pad_token_id(tok, arch),
                          do_sample=False)


def fallback_pred(out, tok, arch, input_len, logit_mode, safe_id, unsafe_id):
    if logit_mode:
        logits = out.logits[:, -1, :]
        return 1 if float(logits[0, unsafe_id]) > float(logits[0, safe_id]) else 0
    text = sce.decode_output(tok, arch, out, input_len)
    return sce.text_to_pred(text, arch)


def extract_states(model, tok, arch, prompts, layer_list, newline_id,
                   tag, key, cache_dir, save_cache=False):
    layer_str = "_".join(str(l) for l in layer_list)
    emb_path  = os.path.join(cache_dir, f"{key}_{tag}_eexit_layers{layer_str}.npz")
    if save_cache and os.path.exists(emb_path):
        data = np.load(emb_path)
        if data["valid_mask"].shape[0] == len(prompts):
            print(f"  [cache] {emb_path}")
            return {li: data[f"layer_{li}"] for li in layer_list}, data["valid_mask"]
        print("  [cache] Size mismatch — recomputing"); os.remove(emb_path)

    device     = next(model.parameters()).device
    d_model    = sce.get_hidden_size(model, arch)
    all_layers = sce.get_layers(model, arch)
    states     = {li: np.zeros((len(prompts), d_model), dtype=np.float32)
                  for li in layer_list}
    valid_mask = np.ones(len(prompts), dtype=bool)

    for i in tqdm(range(len(prompts)), desc=f"  {tag} states", ncols=72):
        inp = build_exit_input(prompts[i], tok, arch, newline_id)
        if inp["input_ids"].shape[1] > sce.MAX_SEQ_LEN:
            valid_mask[i] = False; continue
        inp = sce.to_device(inp, device)
        buf = {}

        def make_hook(li):
            def _hook(mod, _, output):
                h = output[0] if isinstance(output, tuple) else output
                buf[li] = h[:, -1, :].squeeze(0).float().cpu().numpy()
            return _hook

        handles = [all_layers[li].register_forward_hook(make_hook(li))
                   for li in layer_list]
        try:
            with torch.no_grad():
                model(**inp, logits_to_keep=1)
        finally:
            for h in handles: h.remove()
        if len(buf) < len(layer_list):
            valid_mask[i] = False; continue
        for li in layer_list:
            states[li][i] = buf[li]

    if save_cache:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(emb_path, valid_mask=valid_mask,
                 **{f"layer_{li}": states[li] for li in layer_list})
        print(f"  Saved → {emb_path}")
    return states, valid_mask


def layer_outputs(states, verdict_fn, safe_id, unsafe_id, device,
                  dtype=torch.bfloat16, chunk=256):
    out = {}
    for li, mat in states.items():
        n = mat.shape[0]
        verd  = np.zeros(n, dtype=np.int8)
        smax  = np.zeros(n, dtype=np.float32)
        sdiff = np.zeros(n, dtype=np.float32)
        vmarg = np.zeros(n, dtype=np.float32)
        for i in range(0, n, chunk):
            h      = torch.tensor(mat[i:i + chunk], device=device, dtype=dtype)
            logits = verdict_fn(h)
            probs  = torch.softmax(logits, dim=-1)
            top2   = torch.topk(probs, 2, dim=-1).values
            verd[i:i + chunk]  = (logits[:, unsafe_id] > logits[:, safe_id]) \
                                     .to(torch.int8).cpu().numpy()
            smax[i:i + chunk]  = top2[:, 0].cpu().numpy()
            sdiff[i:i + chunk] = (top2[:, 0] - top2[:, 1]).cpu().numpy()
            vmarg[i:i + chunk] = (probs[:, unsafe_id] - probs[:, safe_id]) \
                                     .abs().cpu().numpy()
        out[li] = {"verdict": verd, "softmax_max": smax,
                   "softmax_diff": sdiff, "verdict_margin": vmarg}
    return out


def train_meta_heads(states, louts, full_verd, monitored, device, shared=False):
    from sklearn.linear_model import LogisticRegression

    if shared:
        X = np.concatenate([states[li] for li in monitored], axis=0)
        y = np.concatenate([(louts[li]["verdict"] == full_verd).astype(np.int32)
                            for li in monitored])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        w = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
        b = torch.tensor(float(clf.intercept_[0]), device=device)
        print(f"    shared head: n={len(y):,} (pooled over {len(monitored)} "
              f"layers)  consistency={y.mean():.4f}  train_acc={clf.score(X, y):.4f}")
        return {li: (w, b) for li in monitored}

    heads = {}
    for li in monitored:
        X = states[li]
        y = (louts[li]["verdict"] == full_verd).astype(np.int32)
        if len(np.unique(y)) < 2:
            p = float(y.mean())
            w = torch.zeros(X.shape[1], device=device)
            b = torch.tensor(math.log(max(p, 1e-6) / max(1 - p, 1e-6)),
                             device=device)
            heads[li] = (w, b)
            print(f"    layer {li:>3}: degenerate (consistency={p:.4f}) — constant head")
            continue
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(X, y)
        heads[li] = (
            torch.tensor(clf.coef_[0], dtype=torch.float32, device=device),
            torch.tensor(float(clf.intercept_[0]), device=device),
        )
        acc = clf.score(X, y)
        print(f"    layer {li:>3}: n={len(y):,}  consistency={y.mean():.4f}  "
              f"train_acc={acc:.4f}")
    return heads


def build_conf_matrix(method, states, louts, monitored, meta_heads=None):
    n = next(iter(states.values())).shape[0]
    C = np.zeros((n, len(monitored)), dtype=np.float32)
    for j, li in enumerate(monitored):
        if method in ("softmax_diff", "softmax_max", "verdict_margin"):
            C[:, j] = louts[li][method]
        elif method == "state":
            if j == 0:
                C[:, j] = -1.0
            else:
                a, b = states[monitored[j - 1]], states[li]
                num  = np.sum(a * b, axis=1)
                den  = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
                C[:, j] = num / den
        elif method == "meta":
            w, b = meta_heads[li]
            z = states[li] @ w.cpu().numpy() + float(b)
            C[:, j] = 1.0 / (1.0 + np.exp(-z))
    return C


def hb_pvalue(emp_risk, n, delta):
    from scipy.stats import binom

    def h1(a, b):
        eps = 1e-12
        a = min(max(a, eps), 1 - eps)
        b = min(max(b, eps), 1 - eps)
        return a * math.log(a / b) + (1 - a) * math.log((1 - a) / (1 - b))

    p_h = math.exp(-n * h1(min(emp_risk, delta), delta))
    p_b = math.e * binom.cdf(math.ceil(n * emp_risk), n, delta)
    return float(min(p_h, p_b, 1.0))


def simulate_exits(C, V, full_verd, lam, monitored, n_layers):
    hits  = C >= lam
    has   = hits.any(axis=1)
    first = np.argmax(hits, axis=1)
    idx   = np.arange(len(full_verd))
    pred  = np.where(has, V[idx, first], full_verd)
    risk  = float(np.mean(pred != full_verd))
    used  = np.where(has, np.asarray(monitored)[first] + 1, n_layers)
    return risk, float(used.mean()), float(has.mean())


def ltt_calibrate(C, V, full_verd, monitored, n_layers, delta,
                  epsilon=LTT_EPSILON, grid=LAMBDA_GRID, verbose=True):
    n        = len(full_verd)
    lam_star = 1.0
    risk_star = 0.0
    rows     = []
    for lam in grid:
        risk, avg_layers, exit_rate = simulate_exits(
            C, V, full_verd, lam, monitored, n_layers)
        p      = hb_pvalue(risk, n, delta)
        reject = p <= epsilon
        rows.append((lam, risk, p, avg_layers, exit_rate, reject))
        if reject:
            lam_star, risk_star = float(lam), risk
        else:
            break
    if verbose:
        print(f"    {'lambda':>7} {'risk':>8} {'p-value':>10} "
              f"{'avg_layers':>11} {'exit%':>7}  decision")
        for lam, risk, p, al, er, rej in rows:
            print(f"    {lam:>7.2f} {risk:>8.4f} {p:>10.2e} "
                  f"{al:>11.2f} {er * 100:>6.1f}%  "
                  f"{'reject (valid)' if rej else 'accept (stop)'}")
        print(f"    → lambda* = {lam_star:.2f}  (delta={delta}, "
              f"epsilon={epsilon}, n={n}, calib risk={risk_star:.4f})")
    return lam_star, risk_star


def infer_early_exit(model, tok, arch, prompt, all_layers, monitored, lam,
                     conf_method, verdict_fn, meta_heads, newline_id,
                     safe_id, unsafe_id, logit_mode):
    inp = build_exit_input(prompt, tok, arch, newline_id)
    if inp["input_ids"].shape[1] > sce.MAX_SEQ_LEN:
        return 0, "lg", {}, None
    device    = next(model.parameters()).device
    inp       = sce.to_device(inp, device)
    input_len = inp["input_ids"].shape[1]
    state     = {"decided": False, "prev": None, "confs": {}, "exit_layer": None}
    last      = monitored[-1]

    def make_hook(li):
        def _hook(mod, _, output):
            if state["decided"]:
                return
            h = output[0] if isinstance(output, tuple) else output
            h = h[:, -1, :]
            logits = None
            if conf_method in ("softmax_diff", "softmax_max", "verdict_margin"):
                logits = verdict_fn(h)
                probs  = torch.softmax(logits, dim=-1)
                if conf_method == "softmax_max":
                    conf = float(probs.max())
                elif conf_method == "softmax_diff":
                    top2 = torch.topk(probs, 2, dim=-1).values[0]
                    conf = float(top2[0] - top2[1])
                else:
                    conf = float((probs[0, unsafe_id] - probs[0, safe_id]).abs())
            elif conf_method == "state":
                prev = state["prev"]
                state["prev"] = h
                conf = (float(F.cosine_similarity(h.float(), prev.float(), dim=-1))
                        if prev is not None else -1.0)
            else:
                w, b = meta_heads[li]
                conf = float(torch.sigmoid(h.float().squeeze(0) @ w + b))
            state["confs"][li] = conf

            if conf >= lam:
                if logits is None:
                    logits = verdict_fn(h)
                pred = 1 if float(logits[0, unsafe_id]) > float(logits[0, safe_id]) else 0
                state["decided"]    = True
                state["exit_layer"] = li
                raise _EarlyExit(pred)
            if li == last:
                state["decided"] = True
        return _hook

    handles   = [all_layers[li].register_forward_hook(make_hook(li))
                 for li in monitored]
    exit_hit  = False
    out       = None
    try:
        with torch.no_grad():
            out = guard_call(model, inp, tok, arch, logit_mode)
    except _EarlyExit as e:
        exit_hit  = True
        exit_pred = e.val
        if not logit_mode and hasattr(model, "_cache"):
            del model._cache
    finally:
        for h in handles: h.remove()

    if exit_hit:
        path = "exit_harm" if exit_pred == 1 else "exit_less"
        return exit_pred, path, state["confs"], state["exit_layer"]
    pred = fallback_pred(out, tok, arch, input_len, logit_mode, safe_id, unsafe_id)
    return pred, "lg", state["confs"], None


def run_early_exit(model, tok, arch, test_prompts, test_labels, monitored, lam,
                   conf_method, verdict_fn, meta_heads, newline_id,
                   safe_id, unsafe_id, logit_mode, n_layers):
    all_layers = sce.get_layers(model, arch)

    for p in test_prompts[:sce.WARMUP_N]:
        infer_early_exit(model, tok, arch, p, all_layers, monitored, lam,
                         conf_method, verdict_fn, meta_heads, newline_id,
                         safe_id, unsafe_id, logit_mode)
    torch.cuda.synchronize()

    eh_preds, eh_true, eh_times = [], [], []
    el_preds, el_true, el_times = [], [], []
    lg_preds, lg_true, lg_times = [], [], []
    all_preds, all_times        = [], []
    exit_counts = {li: 0 for li in monitored}
    exit_counts["lg"] = 0
    layers_used = []
    records     = []

    for i, (p, lbl) in enumerate(tqdm(zip(test_prompts, test_labels),
                                      total=len(test_prompts),
                                      desc="  early-exit", ncols=72)):
        t0 = time.perf_counter()
        pred, path, confs, exit_layer = infer_early_exit(
            model, tok, arch, p, all_layers, monitored, lam,
            conf_method, verdict_fn, meta_heads, newline_id,
            safe_id, unsafe_id, logit_mode)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        all_preds.append(pred); all_times.append(t)
        exit_counts[exit_layer if exit_layer is not None else "lg"] += 1
        layers_used.append(exit_layer + 1 if exit_layer is not None else n_layers)
        records.append({"idx": i, "true_label": int(lbl), "pred": int(pred),
                        "correct": int(pred) == int(lbl), "latency_ms": t * 1000,
                        "path": path,
                        "exit_layer": exit_layer if exit_layer is not None else "lg",
                        "prompt": p})

        if path == "exit_harm":
            eh_preds.append(pred); eh_true.append(lbl); eh_times.append(t)
        elif path == "exit_less":
            el_preds.append(pred); el_true.append(lbl); el_times.append(t)
        else:
            lg_preds.append(pred); lg_true.append(lbl); lg_times.append(t)

    total = len(test_prompts)
    print(f"\n  Exit layer summary:")
    print(f"  {'─' * 36}")
    for li in monitored:
        n = exit_counts[li]
        print(f"  Layer {li:>3} : {n:>6,}  ({n / total * 100:5.1f}%)")
    n_lg = exit_counts["lg"]
    print(f"  {'full model':>10} : {n_lg:>6,}  ({n_lg / total * 100:5.1f}%)")
    print(f"  Avg decoder layers used : {np.mean(layers_used):.2f} / {n_layers}")
    print(f"  {'─' * 36}")

    return {
        "exit_harm":   sce._metrics(eh_true, eh_preds, eh_times),
        "exit_less":   sce._metrics(el_true, el_preds, el_times),
        "lg":          sce._metrics(lg_true, lg_preds, lg_times),
        "combined":    sce._metrics(list(test_labels), all_preds, all_times),
        "exit_counts": exit_counts,
        "avg_layers":  float(np.mean(layers_used)),
        "_records":    records,
    }


def print_model_result(label, monitored, lam, conf_method, res):
    eh, el, lg, com = (res["exit_harm"], res["exit_less"],
                       res["lg"], res["combined"])
    n = com["n"]
    print(f"\n{'=' * 74}")
    print(f"  {label}  (early exit: {conf_method}, lambda={lam:.2f}, "
          f"layers={monitored})")
    print(f"{'=' * 74}")

    print(f"\n  {'':32} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  {'n':>6}")
    print(f"  {'─' * 68}")
    for name, r in [("Early-exit harmful path",  eh),
                    ("Early-exit harmless path", el),
                    ("Full-model path",          lg),
                    ("─" * 32,                   None),
                    ("Combined",                 com)]:
        if r is None:
            print(f"  {name}"); continue
        print(f"  {name:<32} {r['acc']:7.4f} {r['prec']:7.4f} "
              f"{r['rec']:7.4f} {r['f1']:7.4f}  {r['n']:>6,}")

    print(f"\n  {'':32} {'timing':>9}")
    print(f"  {'─' * 64}")
    for name, r in [("Early-exit harmful path",  eh),
                    ("Early-exit harmless path", el),
                    ("Full-model path",          lg),
                    ("─" * 32,                   None),
                    ("Combined",                 com)]:
        if r is None:
            print(f"  {name}"); continue
        print(f"  {name:<32} avg={r['ms_avg']:>7.1f}ms "
              f"min={r['ms_min']:>7.1f}ms max={r['ms_max']:>7.1f}ms  "
              f"harm={r['ms_harmful']:>7.1f}ms harml={r['ms_harmless']:>7.1f}ms")

    pct = lambda r: r["n"] / n * 100 if n > 0 else 0
    print(f"\n  Early exit (harmful)  : {eh['n']:,} ({pct(eh):.1f}%)  "
          f"[harmful={eh['n_harmful']:,}  harmless={eh['n_harmless']:,}]")
    print(f"  Early exit (harmless) : {el['n']:,} ({pct(el):.1f}%)  "
          f"[harmful={el['n_harmful']:,}  harmless={el['n_harmless']:,}]")
    print(f"  Full-model fallback   : {lg['n']:,} ({pct(lg):.1f}%)  "
          f"[harmful={lg['n_harmful']:,}  harmless={lg['n_harmless']:,}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-dataset", choices=list(sce.DATASET_PATHS.keys()),
                        nargs="+", default=["wildguardmix"])
    parser.add_argument("--cache-dataset", choices=list(sce.DATASET_PATHS.keys()),
                        nargs="+", default=None,
                        help="Dataset(s) whose calibration split calibrates lambda "
                             "(same split the semantic cache uses for thresholds)")
    parser.add_argument("--conf-method", nargs="+", choices=list(CONF_METHODS),
                        default=["softmax_diff"],
                        help="CALM confidence measure(s) to sweep")
    parser.add_argument("--delta", type=float, nargs="+", default=[0.05],
                        help="LTT consistency tolerance(s) to sweep "
                             "(P[disagree with full model] <= delta)")
    parser.add_argument("--conf-threshold", type=float, nargs="+", default=None,
                        help="Fixed lambda value(s); skips LTT calibration")
    parser.add_argument("--epsilon", type=float, default=LTT_EPSILON,
                        help="LTT FWER level (default 0.05)")
    parser.add_argument("--exit-layers", type=int, nargs="+", default=None,
                        help="Monitored exit layers (default: range(5, n_layers, 5), "
                             "same as semantic-cache --layerwise)")
    parser.add_argument("--first-exit", type=int, default=None,
                        help="First exit layer (t5x calm repo: first_exit; paper "
                             "uses 0). Combined with --exit-interval, monitored "
                             "layers = range(first_exit, n_layers, exit_interval)")
    parser.add_argument("--exit-interval", type=int, default=None,
                        help="Layers between exit checks (t5x calm repo: "
                             "exit_interval; paper uses 1 = every layer)")
    parser.add_argument("--meta-shared", action="store_true",
                        help="Train a single meta classifier shared across all "
                             "layers (paper Sec. 5) instead of one per layer")
    parser.add_argument("--meta-train-limit", type=int, default=META_MAX_TRAIN,
                        help="Max index-split prompts used to train the meta "
                             "early-exit classifier")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--calib-limit", type=int, default=None,
                        help="Subsample the calibration split to at most N "
                             "prompts (seeded); mainly for quick runs")
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--logit", action="store_true",
                        help="Single-forward fallback (read safe/unsafe logits) "
                             "instead of autoregressive generation — same as "
                             "semantic_cache_eval --logit")
    parser.add_argument("--cache-emb", action="store_true",
                        help="Cache calibration hidden states to disk")
    parser.add_argument("--skip-lg-only", action="store_true")
    parser.add_argument("--per-example-log", action="store_true")
    parser.add_argument("--summary-file", type=str, default=None)
    args = parser.parse_args()

    sce.set_seed()
    cfg           = sce.resolve_model_cfg(args.model)
    calib_ds_list = args.cache_dataset if args.cache_dataset else args.test_dataset
    fixed_mode    = args.conf_threshold is not None

    print("=" * 74)
    print("  CALM Early-Exit Baseline  (vs. security-aware semantic cache)")
    print(f"  Model          : {args.model}")
    print(f"  Calib dataset  : {calib_ds_list}")
    print(f"  Test  dataset  : {args.test_dataset}")
    print(f"  Conf methods   : {args.conf_method}")
    if fixed_mode:
        print(f"  Fixed lambdas  : {args.conf_threshold}")
    else:
        print(f"  LTT delta sweep: {args.delta}  (epsilon={args.epsilon})")
    print(f"  Fallback       : {'logit single-forward' if args.logit else 'generation'}")
    print("=" * 74)

    test_data = {}
    for td in args.test_dataset:
        test_path = args.test_path if args.test_path else sce.DATASET_PATHS[td]["test"]
        tp, tl = sce.load_data(test_path, td, limit=args.limit)
        test_data[td] = (tp, tl)
        print(f"  test [{td}]: {len(tp):,}  "
              f"(harmful={tl.sum():,}  harmless={(tl == 0).sum():,})")

    index_prompts, index_labels = [], []
    calib_prompts, calib_labels = [], []
    calib_cache_dir = None
    for ds in calib_ds_list:
        train_path = sce.DATASET_PATHS[ds]["cache_data"]
        if train_path is None:
            raise ValueError(
                f"Dataset '{ds}' has no cache split — pass --cache-dataset "
                f"with a dataset that has one (as semantic_cache_eval does).")
        if calib_cache_dir is None:
            calib_cache_dir = sce.DATASET_PATHS[ds]["cache"]
        tp, tl = sce.load_data(train_path, ds)
        tp, tl, n_removed = sce.dedup(tp, tl)
        print(f"\n  [{ds}] cache (after dedup): {len(tp):,}  (removed {n_removed:,}  "
              f"harmful={tl.sum():,}  harmless={(tl == 0).sum():,})")
        idx_mask, cal_mask = sce.stratified_split(tl)
        index_prompts += [p for p, m in zip(tp, idx_mask) if m]
        index_labels.append(tl[idx_mask])
        calib_prompts += [p for p, m in zip(tp, cal_mask) if m]
        calib_labels.append(tl[cal_mask])
    index_labels = np.concatenate(index_labels)
    calib_labels = np.concatenate(calib_labels)
    if args.calib_limit and len(calib_prompts) > args.calib_limit:
        rng = np.random.default_rng(sce.CALIB_SEED)
        sel = np.sort(rng.choice(len(calib_prompts), args.calib_limit, replace=False))
        calib_prompts = [calib_prompts[i] for i in sel]
        calib_labels  = calib_labels[sel]
        print(f"  Calib split subsampled to {len(calib_prompts):,} (--calib-limit)")
    print(f"\n  Combined calib split : {len(calib_prompts):,}  "
          f"(harmful={calib_labels.sum():,}  harmless={(calib_labels == 0).sum():,})")

    print(f"\n  Loading model ...")
    model, tok = sce.load_model(cfg)
    arch     = cfg["arch"]
    device   = next(model.parameters()).device
    n_layers = len(sce.get_layers(model, arch))
    print(f"  num_layers={n_layers}  hidden_size={sce.get_hidden_size(model, arch)}")

    if args.exit_layers:
        monitored = sorted(args.exit_layers)
    elif args.first_exit is not None or args.exit_interval is not None:
        monitored = list(range(args.first_exit or 0, n_layers,
                               args.exit_interval or 1))
    else:
        monitored = list(range(5, n_layers, 5))
    assert all(0 <= li < n_layers for li in monitored), \
        f"exit layers must be in [0, {n_layers})"
    collect_layers = sorted(set(monitored) | {n_layers - 1})
    print(f"  Monitored exit layers: {monitored}")

    safe_id, unsafe_id = sce.get_class_token_ids(tok, arch)
    newline_id = resolve_newline_id(tok, arch, is_finetuned=cfg["ft"] is not None)
    verdict_fn = get_verdict_head(model, arch)
    print(f"  Verdict tokens: safe_id={safe_id}  unsafe_id={unsafe_id}  "
          f"newline_id={newline_id}")

    if args.logit:
        sce.LOGIT_MODE  = True
        sce.SAFE_ID     = safe_id
        sce.UNSAFE_ID   = unsafe_id
        sce._NEWLINE_ID = newline_id
        print(f"  Classification : logit-based single forward")
    else:
        sce.LOGIT_MODE = False
        print(f"  Classification : generation-based (max_new_tokens=20)")

    print(f"\n  Extracting calibration hidden states "
          f"(layers {collect_layers}) ...")
    cal_states, cal_valid = extract_states(
        model, tok, arch, calib_prompts, collect_layers, newline_id,
        "calib", cfg["key"], calib_cache_dir, save_cache=args.cache_emb)
    cal_states = {li: m[cal_valid] for li, m in cal_states.items()}
    print(f"  Valid calibration samples: {int(cal_valid.sum()):,}")

    print(f"  Computing per-layer verdicts/confidences ...")
    cal_louts = layer_outputs(cal_states, verdict_fn, safe_id, unsafe_id, device)
    full_verd = cal_louts[n_layers - 1]["verdict"]
    V = np.stack([cal_louts[li]["verdict"] for li in monitored], axis=1)

    print(f"\n  Per-layer consistency with the full model (calibration):")
    for li in monitored:
        cons = float(np.mean(cal_louts[li]["verdict"] == full_verd))
        print(f"    layer {li:>3}: {cons:.4f}")

    meta_heads = None
    if "meta" in args.conf_method:
        rng = np.random.default_rng(sce.SEED)
        if len(index_prompts) > args.meta_train_limit:
            sel = rng.choice(len(index_prompts), args.meta_train_limit, replace=False)
            meta_prompts = [index_prompts[i] for i in sorted(sel)]
        else:
            meta_prompts = index_prompts
        print(f"\n  Extracting meta-training hidden states "
              f"({len(meta_prompts):,} index-split prompts) ...")
        idx_states, idx_valid = extract_states(
            model, tok, arch, meta_prompts, collect_layers, newline_id,
            "meta_train", cfg["key"], calib_cache_dir, save_cache=args.cache_emb)
        idx_states = {li: m[idx_valid] for li, m in idx_states.items()}
        idx_louts  = layer_outputs(idx_states, verdict_fn, safe_id, unsafe_id, device)
        idx_full   = idx_louts[n_layers - 1]["verdict"]
        print(f"  Training early-exit classifier heads "
              f"({'shared' if args.meta_shared else 'per-layer'}) ...")
        meta_heads = train_meta_heads(idx_states, idx_louts, idx_full,
                                      monitored, device, shared=args.meta_shared)
        del idx_states, idx_louts

    lg_res_by_ds = {}
    if not args.skip_lg_only:
        for td, (tp, tl) in test_data.items():
            print(f"\n  LG-only baseline [{td}] ...")
            lg_res_by_ds[td] = sce.run_lg_only(model, tok, arch, tp, tl)
            if args.per_example_log:
                sce.save_per_example_log(
                    lg_res_by_ds[td].get("_records", []),
                    os.path.join(sce.TOTAL_LOG_DIR,
                                 f"{cfg['label']}_{td}_lg_only.csv"))

    sweep_vals = args.conf_threshold if fixed_mode else args.delta
    combos = list(itertools.product(args.conf_method, sweep_vals))
    print(f"\n  Running {len(combos)} combo(s) × {len(args.test_dataset)} "
          f"test dataset(s) ...")

    all_rows_by_ds = {td: [] for td in args.test_dataset}

    for method, val in combos:
        tag = (f"method={method}  lambda={val}" if fixed_mode
               else f"method={method}  delta={val}")
        print(f"\n{'#' * 74}\n  {tag}\n{'#' * 74}")

        C = build_conf_matrix(method, cal_states, cal_louts, monitored,
                              meta_heads=meta_heads)
        if fixed_mode:
            lam = float(val)
            calib_risk, calib_layers, calib_exit = simulate_exits(
                C, V, full_verd, lam, monitored, n_layers)
            print(f"  Fixed lambda={lam:.2f}: calib risk={calib_risk:.4f}  "
                  f"avg_layers={calib_layers:.2f}  exit%={calib_exit * 100:.1f}")
        else:
            print(f"  LTT calibration (delta={val}, epsilon={args.epsilon}):")
            lam, calib_risk = ltt_calibrate(
                C, V, full_verd, monitored, n_layers, val, epsilon=args.epsilon)

        for td, (test_prompts, test_labels) in test_data.items():
            lg_res = lg_res_by_ds.get(td)
            print(f"\n  Evaluating [{td}] ...")
            res = run_early_exit(
                model, tok, arch, test_prompts, test_labels, monitored, lam,
                method, verdict_fn, meta_heads, newline_id,
                safe_id, unsafe_id, args.logit, n_layers)
            label = f"{cfg['label']} [early-exit]"
            print_model_result(label, monitored, lam, method, res)

            sweep_sfx = (f"{method}_lam{val}" if fixed_mode
                         else f"{method}_delta{val}")
            if args.per_example_log:
                sce.save_per_example_log(
                    res.get("_records", []),
                    os.path.join(sce.TOTAL_LOG_DIR,
                                 f"{cfg['label']}_{td}_earlyexit_{sweep_sfx}.csv"))

            combined = res["combined"]
            total    = combined["n"] if combined["n"] > 0 else 1
            row = {
                "conf_method":   method,
                "delta":         "-" if fixed_mode else val,
                "lambda":        lam,
                "calib_risk":    round(calib_risk, 4),
                "acc":           combined["acc"],
                "prec":          combined["prec"],
                "rec":           combined["rec"],
                "f1":            combined["f1"],
                "ms_avg":        combined["ms_avg"],
                "ms_min":        combined["ms_min"],
                "ms_max":        combined["ms_max"],
                "avg_layers":    round(res["avg_layers"], 2),
                "exit_harm_n":   res["exit_harm"]["n"],
                "exit_harm_pct": round(res["exit_harm"]["n"] / total * 100, 1),
                "exit_less_n":   res["exit_less"]["n"],
                "exit_less_pct": round(res["exit_less"]["n"] / total * 100, 1),
                "lg_n":          res["lg"]["n"],
                "lg_pct":        round(res["lg"]["n"] / total * 100, 1),
            }
            for li, cnt in res["exit_counts"].items():
                key = f"layer_{li}" if li != "lg" else "layer_lg"
                row[key + "_n"]   = cnt
                row[key + "_pct"] = round(cnt / total * 100, 1)
            row["lg_only_acc"]    = lg_res["acc"]    if lg_res else 0
            row["lg_only_prec"]   = lg_res["prec"]   if lg_res else 0
            row["lg_only_rec"]    = lg_res["rec"]    if lg_res else 0
            row["lg_only_f1"]     = lg_res["f1"]     if lg_res else 0
            row["lg_only_ms"]     = lg_res["ms_avg"] if lg_res else 0
            row["lg_only_ms_min"] = lg_res["ms_min"] if lg_res else 0
            row["lg_only_ms_max"] = lg_res["ms_max"] if lg_res else 0
            all_rows_by_ds[td].append(row)

    sweep_col      = "lambda" if fixed_mode else "delta"
    summary_prefix = args.summary_file
    if summary_prefix and summary_prefix.endswith(".csv"):
        summary_prefix = summary_prefix[:-4]

    for td in args.test_dataset:
        rows   = all_rows_by_ds[td]
        lg_res = lg_res_by_ds.get(td)
        W = 74
        print(f"\n\n{'=' * W}")
        print(f"  SWEEP SUMMARY — early exit — test dataset: {td}")
        print(f"{'=' * W}")
        print(f"  {'method':<15} {sweep_col:<7} {'lam':>5} {'Acc':>7} {'Prec':>7} "
              f"{'Rec':>7} {'F1':>7}  {'ms':>7} {'layers':>6} {'exit%':>6}")
        print(f"  {'─' * 70}")
        for r in rows:
            exit_pct = r["exit_harm_pct"] + r["exit_less_pct"]
            print(f"  {r['conf_method']:<15} {str(r['delta']):<7} {r['lambda']:>5.2f} "
                  f"{r['acc']:7.4f} {r['prec']:7.4f} {r['rec']:7.4f} {r['f1']:7.4f}  "
                  f"{r['ms_avg']:6.1f}ms {r['avg_layers']:>6.2f} {exit_pct:>5.1f}%")
        if lg_res:
            print(f"\n  LG-only baseline: acc={lg_res['acc']:.4f}  "
                  f"prec={lg_res['prec']:.4f}  rec={lg_res['rec']:.4f}  "
                  f"f1={lg_res['f1']:.4f}  avg={lg_res['ms_avg']:.1f}ms "
                  f"min={lg_res['ms_min']:.1f}ms max={lg_res['ms_max']:.1f}ms")
        print(f"{'=' * W}")

        if args.summary_file and rows:
            out_path = (args.summary_file if len(args.test_dataset) == 1
                        else f"{summary_prefix}_{td}.csv")
            fieldnames = list(rows[0].keys())
            with open(out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                if lg_res:
                    lg_row = {k: 0 for k in fieldnames}
                    lg_row["conf_method"]    = "lg_only"
                    lg_row["delta"]          = "-"
                    lg_row["acc"]            = lg_res["acc"]
                    lg_row["prec"]           = lg_res["prec"]
                    lg_row["rec"]            = lg_res["rec"]
                    lg_row["f1"]             = lg_res["f1"]
                    lg_row["ms_avg"]         = lg_res["ms_avg"]
                    lg_row["ms_min"]         = lg_res["ms_min"]
                    lg_row["ms_max"]         = lg_res["ms_max"]
                    lg_row["avg_layers"]     = n_layers
                    lg_row["lg_only_acc"]    = lg_res["acc"]
                    lg_row["lg_only_prec"]   = lg_res["prec"]
                    lg_row["lg_only_rec"]    = lg_res["rec"]
                    lg_row["lg_only_f1"]     = lg_res["f1"]
                    lg_row["lg_only_ms"]     = lg_res["ms_avg"]
                    lg_row["lg_only_ms_min"] = lg_res["ms_min"]
                    lg_row["lg_only_ms_max"] = lg_res["ms_max"]
                    writer.writerow(lg_row)
                writer.writerows(rows)
            print(f"\n  Summary saved → {out_path}")

    sce.unload(model)


if __name__ == "__main__":
    main()
