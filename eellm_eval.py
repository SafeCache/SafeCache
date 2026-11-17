import argparse
import csv
import os
import time

import numpy as np
import torch
from tqdm import tqdm

import semantic_cache_eval as sce
from semantic_cache_eval import _EarlyExit
from jb_mutate import JB_DATASETS as _JB_DATASETS, load_jb_dataset
import early_exit_eval as eee
import eellm_exit as ee


def infer_eellm(model, tok, arch, prompt, all_layers, heads, thres,
                newline_id, safe_id, unsafe_id, logit_mode):
    inp = eee.build_exit_input(prompt, tok, arch, newline_id)
    if inp["input_ids"].shape[1] > sce.MAX_SEQ_LEN:
        return 0, "lg", {}, None, 0, 0
    device    = next(model.parameters()).device
    inp       = sce.to_device(inp, device)
    input_len = inp["input_ids"].shape[1]
    exit_layers = sorted(heads)
    state = {"decided": False, "confs": {}, "exit_layer": None}
    last  = exit_layers[-1]

    def make_hook(li):
        head = heads[li]

        def _hook(mod, _, output):
            if state["decided"]:
                return
            h = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                logits = ee.exit_logits_last(head, h).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            conf = float(log_probs.max().exp())
            state["confs"][li] = conf
            if conf >= thres:
                pred = 1 if float(logits[0, unsafe_id]) > float(logits[0, safe_id]) else 0
                state["decided"]    = True
                state["exit_layer"] = li
                raise _EarlyExit(pred)
            if li == last:
                state["decided"] = True
        return _hook

    handles = ([] if thres >= 1.0 else
               [all_layers[li].register_forward_hook(make_hook(li))
                for li in exit_layers])
    exit_hit = False
    out      = None
    try:
        with torch.no_grad():
            out = eee.guard_call(model, inp, tok, arch, logit_mode)
    except _EarlyExit as e:
        exit_hit  = True
        exit_pred = e.val
        if not logit_mode and hasattr(model, "_cache"):
            del model._cache
    finally:
        for h in handles:
            h.remove()

    if exit_hit:
        path = "exit_harm" if exit_pred == 1 else "exit_less"
        return exit_pred, path, state["confs"], state["exit_layer"], input_len, 0
    pred = eee.fallback_pred(out, tok, arch, input_len, logit_mode,
                             safe_id, unsafe_id)
    n_gen = 0 if logit_mode else max(0, out.shape[1] - input_len)
    return pred, "lg", state["confs"], None, input_len, n_gen


def infer_eellm_pass(model, tok, arch, prompt, all_layers, heads, exit_at,
                     newline_id, safe_id, unsafe_id, logit_mode):
    inp = eee.build_exit_input(prompt, tok, arch, newline_id)
    if inp["input_ids"].shape[1] > sce.MAX_SEQ_LEN:
        return {}, {}, 0, True, 0, 0
    device    = next(model.parameters()).device
    inp       = sce.to_device(inp, device)
    input_len = inp["input_ids"].shape[1]
    exit_layers = sorted(heads)
    state = {"decided": False, "confs": {}, "preds": {}}
    last  = exit_layers[-1]

    def make_hook(li):
        head = heads[li]

        def _hook(mod, _, output):
            if state["decided"]:
                return
            h = output[0] if isinstance(output, tuple) else output
            with torch.no_grad():
                logits = ee.exit_logits_last(head, h).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            state["confs"][li] = float(log_probs.max().exp())
            state["preds"][li] = (
                1 if float(logits[0, unsafe_id]) > float(logits[0, safe_id]) else 0)
            if li == exit_at:
                state["decided"] = True
                raise _EarlyExit(state["preds"][li])
            if li == last and exit_at is None:
                state["decided"] = True
        return _hook

    handles  = [all_layers[li].register_forward_hook(make_hook(li))
                for li in exit_layers]
    exit_hit = False
    out      = None
    try:
        with torch.no_grad():
            out = eee.guard_call(model, inp, tok, arch, logit_mode)
    except _EarlyExit:
        exit_hit = True
        if not logit_mode and hasattr(model, "_cache"):
            del model._cache
    finally:
        for h in handles:
            h.remove()

    full_pred = (None if exit_hit else
                 eee.fallback_pred(out, tok, arch, input_len, logit_mode,
                                   safe_id, unsafe_id))
    n_gen = (0 if (exit_hit or logit_mode)
             else max(0, out.shape[1] - input_len))
    return state["confs"], state["preds"], full_pred, False, input_len, n_gen


def run_eellm_sim(model, tok, arch, prompts, labels, heads, grid,
                  newline_id, safe_id, unsafe_id, logit_mode, n_layers,
                  lg_records, flops_profile=None, dump_path=None,
                  confs_only=False):
    all_layers  = sce.get_layers(model, arch)
    exit_layers = sorted(heads)
    n = len(prompts)

    confs   = {li: np.full(n, -1.0, dtype=np.float64) for li in exit_layers}
    epreds  = {li: np.zeros(n, dtype=np.int64) for li in exit_layers}
    fpred   = np.zeros(n, dtype=np.int64)
    t_path  = {li: np.zeros(n) for li in exit_layers}
    t_full  = np.zeros(n)
    skipped = np.zeros(n, dtype=bool)
    n_tok   = np.zeros(n, dtype=np.int64)
    n_gen   = np.zeros(n, dtype=np.int64)

    for p in prompts[:sce.WARMUP_N]:
        infer_eellm_pass(model, tok, arch, p, all_layers, heads, None,
                         newline_id, safe_id, unsafe_id, logit_mode)
    torch.cuda.synchronize()
    for i, p in enumerate(tqdm(prompts, desc="  sim full  ", ncols=72)):
        t0 = time.perf_counter()
        c, ep, fp, sk, tk, ng = infer_eellm_pass(model, tok, arch, p, all_layers,
                                                 heads, None, newline_id,
                                                 safe_id, unsafe_id, logit_mode)
        torch.cuda.synchronize()
        t_full[i] = time.perf_counter() - t0
        if sk:
            skipped[i], t_full[i] = True, 0.0
            continue
        fpred[i] = fp
        n_tok[i], n_gen[i] = tk, ng
        for li in exit_layers:
            confs[li][i]  = c[li]
            epreds[li][i] = ep[li]

    if dump_path:
        np.savez(dump_path, exit_layers=np.array(exit_layers), labels=np.asarray(labels),
                 fpred=fpred, skipped=skipped, n_tok=n_tok, t_full=t_full,
                 **{f"conf_{li}": confs[li] for li in exit_layers},
                 **{f"epred_{li}": epreds[li] for li in exit_layers})
        print(f"  per-prompt sim dump -> {dump_path}")

    for li in ([] if confs_only else exit_layers):
        for p in prompts[:sce.WARMUP_N]:
            infer_eellm_pass(model, tok, arch, p, all_layers, heads, li,
                             newline_id, safe_id, unsafe_id, logit_mode)
        torch.cuda.synchronize()
        for i, p in enumerate(tqdm(prompts, desc=f"  sim exit@{li:<3}", ncols=72)):
            if skipped[i]:
                continue
            t0 = time.perf_counter()
            infer_eellm_pass(model, tok, arch, p, all_layers, heads, li,
                             newline_id, safe_id, unsafe_id, logit_mode)
            torch.cuda.synchronize()
            t_path[li][i] = time.perf_counter() - t0

    if lg_records:
        lg_lat  = np.array([r["latency_ms"] / 1000 for r in lg_records])
        lg_pred = np.array([r["pred"] for r in lg_records])
        lg_fl   = np.array([r.get("gflops", 0.0) * 1e9 for r in lg_records])
        lg_tk   = np.array([r.get("n_tok", 0) for r in lg_records])
    else:
        lg_lat, lg_pred = t_full, fpred
        lg_fl = lg_tk = None

    fl_path, fl_full, fl_lgonly = {}, None, None
    if flops_profile:
        fl_full = np.array([sce.flops_for_sample(
            flops_profile, int(n_tok[i]), None, int(n_gen[i]), exit_layers, {})
            for i in range(n)], dtype=np.float64)
        fl_path = {li: np.array([sce.flops_for_sample(
            flops_profile, int(n_tok[i]), li, 0, exit_layers, {})
            for i in range(n)], dtype=np.float64) for li in exit_layers}
        if lg_fl is None:
            lg_fl = np.array([sce.flops_for_sample(
                flops_profile, int(n_tok[i]), None, int(n_gen[i]), [], {})
                for i in range(n)], dtype=np.float64)
            lg_tk = n_tok

    labels = np.asarray(labels)
    res_by_thres = {}
    for thres in grid:
        preds = np.where(skipped, 0, fpred if thres < 1.0 else lg_pred)
        times = np.where(skipped, 0.0, t_full if thres < 1.0 else lg_lat)
        if flops_profile:
            flops = np.where(skipped, 0.0, fl_full if thres < 1.0 else lg_fl)
            toks  = np.where(skipped, 0, n_tok if thres < 1.0 else lg_tk)
        exit_of = np.full(n, -1, dtype=np.int64)
        if thres < 1.0:
            for li in reversed(exit_layers):
                hit = (~skipped) & (confs[li] >= thres)
                exit_of[hit] = li
                preds[hit]   = epreds[li][hit]
                times[hit]   = t_path[li][hit]
                if flops_profile:
                    flops[hit] = fl_path[li][hit]
        eh = (exit_of >= 0) & (preds == 1)
        el = (exit_of >= 0) & (preds == 0)
        lg = exit_of < 0
        exit_counts = {li: int((exit_of == li).sum()) for li in exit_layers}
        exit_counts["lg"] = int(lg.sum())
        layers_used = np.where(exit_of >= 0, exit_of + 1, n_layers)
        res_by_thres[thres] = {
            "exit_harm": sce._metrics(list(labels[eh]), list(preds[eh]), list(times[eh])),
            "exit_less": sce._metrics(list(labels[el]), list(preds[el]), list(times[el])),
            "lg":        sce._metrics(list(labels[lg]), list(preds[lg]), list(times[lg])),
            "combined":  sce._metrics(list(labels), list(preds), list(times)),
            "exit_counts": exit_counts,
            "avg_layers":  float(layers_used.mean()),
        }
        if flops_profile:
            res_by_thres[thres]["flops"] = sce._flops_stats(
                flops.tolist(), toks.tolist())
    return res_by_thres


def run_eellm(model, tok, arch, test_prompts, test_labels, heads, thres,
              newline_id, safe_id, unsafe_id, logit_mode, n_layers,
              flops_profile=None):
    all_layers  = sce.get_layers(model, arch)
    exit_layers = sorted(heads)
    monitored   = exit_layers if thres < 1.0 else []

    for p in test_prompts[:sce.WARMUP_N]:
        infer_eellm(model, tok, arch, p, all_layers, heads, thres,
                    newline_id, safe_id, unsafe_id, logit_mode)
    torch.cuda.synchronize()

    eh_preds, eh_true, eh_times = [], [], []
    el_preds, el_true, el_times = [], [], []
    lg_preds, lg_true, lg_times = [], [], []
    all_preds, all_times        = [], []
    exit_counts = {li: 0 for li in exit_layers}
    exit_counts["lg"] = 0
    layers_used = []
    records     = []
    fl_list, tk_list = [], []

    for i, (p, lbl) in enumerate(tqdm(zip(test_prompts, test_labels),
                                      total=len(test_prompts),
                                      desc="  ee-llm    ", ncols=72)):
        t0 = time.perf_counter()
        pred, path, confs, exit_layer, tk, ng = infer_eellm(
            model, tok, arch, p, all_layers, heads, thres,
            newline_id, safe_id, unsafe_id, logit_mode)
        torch.cuda.synchronize()
        t = time.perf_counter() - t0

        fl = (sce.flops_for_sample(flops_profile, tk, exit_layer, ng,
                                   monitored, {})
              if flops_profile else 0)
        fl_list.append(fl); tk_list.append(tk)

        all_preds.append(pred); all_times.append(t)
        exit_counts[exit_layer if exit_layer is not None else "lg"] += 1
        layers_used.append(exit_layer + 1 if exit_layer is not None else n_layers)
        records.append({"idx": i, "true_label": int(lbl), "pred": int(pred),
                        "correct": int(pred) == int(lbl), "latency_ms": t * 1000,
                        "path": path,
                        "exit_layer": exit_layer if exit_layer is not None else "lg",
                        "gflops": round(fl / 1e9, 4), "prompt": p})

        if path == "exit_harm":
            eh_preds.append(pred); eh_true.append(lbl); eh_times.append(t)
        elif path == "exit_less":
            el_preds.append(pred); el_true.append(lbl); el_times.append(t)
        else:
            lg_preds.append(pred); lg_true.append(lbl); lg_times.append(t)

    total = len(test_prompts)
    print(f"\n  Exit layer summary (threshold={thres}):")
    print(f"  {'─' * 36}")
    for li in exit_layers:
        n = exit_counts[li]
        print(f"  Layer {li:>3} : {n:>6,}  ({n / total * 100:5.1f}%)")
    n_lg = exit_counts["lg"]
    print(f"  {'full model':>10} : {n_lg:>6,}  ({n_lg / total * 100:5.1f}%)")
    print(f"  Avg decoder layers used : {np.mean(layers_used):.2f} / {n_layers}")

    res = {
        "exit_harm":   sce._metrics(eh_true, eh_preds, eh_times),
        "exit_less":   sce._metrics(el_true, el_preds, el_times),
        "lg":          sce._metrics(lg_true, lg_preds, lg_times),
        "combined":    sce._metrics(list(test_labels), all_preds, all_times),
        "exit_counts": exit_counts,
        "avg_layers":  float(np.mean(layers_used)),
        "_records":    records,
    }
    if flops_profile:
        res["flops"] = sce._flops_stats(fl_list, tk_list)
    return res


def _make_row(res, thres, lg_res):
    combined = res["combined"]
    total    = combined["n"] if combined["n"] > 0 else 1
    row = {
        "conf_method":   "eellm",
        "delta":         "-",
        "lambda":        thres,
        "calib_risk":    "-",
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
    if res.get("flops"):
        row["gflops_avg"]     = round(res["flops"]["gflops_avg"], 3)
        row["gflops_per_tok"] = round(res["flops"]["gflops_per_tok"], 4)
        if lg_res and lg_res.get("flops"):
            row["lg_only_gflops_avg"]     = round(lg_res["flops"]["gflops_avg"], 3)
            row["lg_only_gflops_per_tok"] = round(lg_res["flops"]["gflops_per_tok"], 4)
            if lg_res["flops"]["gflops_avg"] > 0:
                row["flops_ratio"] = round(
                    res["flops"]["gflops_avg"] / lg_res["flops"]["gflops_avg"], 4)
    row["lg_only_acc"]    = lg_res["acc"]    if lg_res else 0
    row["lg_only_prec"]   = lg_res["prec"]   if lg_res else 0
    row["lg_only_rec"]    = lg_res["rec"]    if lg_res else 0
    row["lg_only_f1"]     = lg_res["f1"]     if lg_res else 0
    row["lg_only_ms"]     = lg_res["ms_avg"] if lg_res else 0
    row["lg_only_ms_min"] = lg_res["ms_min"] if lg_res else 0
    row["lg_only_ms_max"] = lg_res["ms_max"] if lg_res else 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--heads", type=str, default=None,
                    help="Tuned exit-head dir from eellm_tune.py "
                         "(default models/eellm-exits-<model-name>)")
    ap.add_argument("--test-dataset", choices=list(sce.DATASET_PATHS.keys()),
                    nargs="+", default=["wildguardmix"])
    ap.add_argument("--early-exit-thres", type=float, nargs="+",
                    default=list(ee.EELLM_THRESHOLDS),
                    help="Confidence threshold sweep (paper grid: "
                         "1.0 0.9 0.8 0.6 0.4 0.2; 1.0 = exits disabled)")
    ap.add_argument("--sim-step", type=float, default=None,
                    help="Fast fine-grid sweep: measure each sample's paths "
                         "once (1 + n_exits instrumented passes), then "
                         "evaluate thresholds --sim-min..1.0 in this step "
                         "offline. Exact metrics; latency identical up to "
                         "timing noise. Overrides --early-exit-thres; no "
                         "per-threshold example logs.")
    ap.add_argument("--sim-min", type=float, default=0.0,
                    help="Lowest threshold of the --sim-step grid "
                         "(default 0.0)")
    ap.add_argument("--sim-dump", type=str, default=None,
                    help="(sim mode) save per-prompt exit confidences/verdicts to "
                         "<prefix>_{dataset}.npz for offline analysis")
    ap.add_argument("--sim-confs-only", action="store_true",
                    help="(sim mode) skip the per-exit-layer latency passes "
                         "(latencies in the summary are then meaningless)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--test_path", type=str, default=None)
    ap.add_argument("--logit", action="store_true",
                    help="Single-forward fallback instead of generation — "
                         "same as semantic_cache_eval --logit")
    ap.add_argument("--skip-lg-only", action="store_true")
    ap.add_argument("--per-example-log", action="store_true")
    ap.add_argument("--summary-file", type=str, default=None)
    args = ap.parse_args()

    sce.set_seed()
    cfg       = sce.resolve_model_cfg(args.model)
    heads_dir = args.heads or ee.heads_dir_for(args.model)
    if not os.path.exists(os.path.join(heads_dir, "config.json")):
        raise FileNotFoundError(
            f"No tuned EE-LLM exit heads at {heads_dir} — run\n"
            f"  python eellm_tune.py --model {args.model}\nfirst.")

    print("=" * 74)
    print("  EE-LLM Early-Exit Baseline  (vs. security-aware semantic cache)")
    print(f"  Model          : {args.model}")
    print(f"  Exit heads     : {heads_dir}")
    print(f"  Test dataset   : {args.test_dataset}")
    print(f"  Thresholds     : {args.early_exit_thres}")
    print(f"  Fallback       : {'logit single-forward' if args.logit else 'generation'}")
    print("=" * 74)

    test_data = {}
    for td in args.test_dataset:
        if td in _JB_DATASETS:
            tp, tl, _meta = load_jb_dataset(td, sce.load_data, sce.DATASET_PATHS,
                                            limit=args.limit)
        else:
            test_path = args.test_path if args.test_path else sce.DATASET_PATHS[td]["test"]
            tp, tl = sce.load_data(test_path, td, limit=args.limit)
        test_data[td] = (tp, tl)
        print(f"  test [{td}]: {len(tp):,}  "
              f"(harmful={tl.sum():,}  harmless={(tl == 0).sum():,})")

    print(f"\n  Loading model ...")
    model, tok = sce.load_model(cfg)
    arch     = cfg["arch"]
    device   = next(model.parameters()).device
    n_layers = len(sce.get_layers(model, arch))

    heads, meta = ee.load_exit_heads(model, arch, heads_dir, device)
    exit_layers = sorted(heads)
    print(f"  num_layers={n_layers}  exit layers: {exit_layers}  "
          f"(tuned on {meta.get('tuning_data')}, loss={meta.get('loss')})")

    flops_profile = sce.make_flops_profile(model, arch)
    h0 = heads[exit_layers[0]]
    flops_profile["head_flops"], flops_profile["head_flops_per_tok"] = \
        ee.exit_head_flops(h0)
    print(f"  FLOPs profile  : {flops_profile['per_tok_layer']/1e6:.1f} MFLOPs/tok/layer "
          f"× {flops_profile['n_layers']} layers, lm_head {flops_profile['lm_head']/1e6:.1f} M, "
          f"exit-head {flops_profile['head_flops']/1e6:.1f} M")

    safe_id, unsafe_id = sce.get_class_token_ids(tok, arch)
    newline_id = eee.resolve_newline_id(tok, arch, is_finetuned=cfg["ft"] is not None)
    print(f"  Verdict tokens: safe_id={safe_id}  unsafe_id={unsafe_id}  "
          f"newline_id={newline_id}")

    if args.logit:
        sce.LOGIT_MODE  = True
        sce.SAFE_ID     = safe_id
        sce.UNSAFE_ID   = unsafe_id
        sce._NEWLINE_ID = newline_id
    else:
        sce.LOGIT_MODE = False

    lg_res_by_ds = {}
    if not args.skip_lg_only:
        for td, (tp, tl) in test_data.items():
            print(f"\n  LG-only baseline [{td}] ...")
            lg_res_by_ds[td] = sce.run_lg_only(model, tok, arch, tp, tl,
                                               flops_profile=flops_profile)
            if args.per_example_log:
                sce.save_per_example_log(
                    lg_res_by_ds[td].get("_records", []),
                    os.path.join(sce.TOTAL_LOG_DIR,
                                 f"{cfg['label']}_{td}_lg_only.csv"))

    all_rows_by_ds = {td: [] for td in args.test_dataset}
    if args.sim_step:
        grid = [round(v, 4) for v in
                np.arange(args.sim_min, 1.0 + 1e-9, args.sim_step)]
        if grid[-1] != 1.0:
            grid.append(1.0)
        print(f"\n  Simulation sweep: {len(grid)} thresholds "
              f"(step {args.sim_step}) from {1 + len(exit_layers)} "
              f"measured passes per dataset")
        for td, (test_prompts, test_labels) in test_data.items():
            lg_res = lg_res_by_ds.get(td)
            print(f"\n  Measuring paths [{td}] ...")
            res_by_thres = run_eellm_sim(
                model, tok, arch, test_prompts, test_labels, heads, grid,
                newline_id, safe_id, unsafe_id, args.logit, n_layers,
                lg_res.get("_records") if lg_res else None,
                flops_profile=flops_profile,
                dump_path=(f"{args.sim_dump}_{td}.npz" if args.sim_dump else None),
                confs_only=args.sim_confs_only)
            for thres in grid:
                all_rows_by_ds[td].append(
                    _make_row(res_by_thres[thres], thres, lg_res))
    else:
        for thres in args.early_exit_thres:
            print(f"\n{'#' * 74}\n  early_exit_thres={thres}\n{'#' * 74}")
            for td, (test_prompts, test_labels) in test_data.items():
                lg_res = lg_res_by_ds.get(td)
                print(f"\n  Evaluating [{td}] ...")
                res = run_eellm(model, tok, arch, test_prompts, test_labels,
                                heads, thres, newline_id, safe_id, unsafe_id,
                                args.logit, n_layers,
                                flops_profile=flops_profile)
                eee.print_model_result(f"{cfg['label']} [ee-llm]", exit_layers,
                                       thres, "eellm", res)
                if args.per_example_log:
                    sce.save_per_example_log(
                        res.get("_records", []),
                        os.path.join(sce.TOTAL_LOG_DIR,
                                     f"{cfg['label']}_{td}_eellm_thres{thres}.csv"))
                all_rows_by_ds[td].append(_make_row(res, thres, lg_res))

    summary_prefix = args.summary_file
    if summary_prefix and summary_prefix.endswith(".csv"):
        summary_prefix = summary_prefix[:-4]

    for td in args.test_dataset:
        rows   = all_rows_by_ds[td]
        lg_res = lg_res_by_ds.get(td)
        W = 74
        print(f"\n\n{'=' * W}")
        print(f"  SWEEP SUMMARY — EE-LLM — test dataset: {td}")
        print(f"{'=' * W}")
        print(f"  {'thres':>6} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}  "
              f"{'ms':>7} {'layers':>6} {'exit%':>6} {'speedup':>8} {'GF/tok':>8}")
        print(f"  {'─' * 66}")
        for r in rows:
            exit_pct = r["exit_harm_pct"] + r["exit_less_pct"]
            speedup  = (r["lg_only_ms"] / r["ms_avg"]) if r["ms_avg"] else 0
            gft = (f"{r['gflops_per_tok']:8.3f}" if "gflops_per_tok" in r
                   else f"{'-':>8}")
            print(f"  {r['lambda']:>6.2f} {r['acc']:7.4f} {r['prec']:7.4f} "
                  f"{r['rec']:7.4f} {r['f1']:7.4f}  {r['ms_avg']:6.1f}ms "
                  f"{r['avg_layers']:>6.2f} {exit_pct:>5.1f}% {speedup:>7.2f}x {gft}")
        if lg_res:
            fl_str = (f"  GF/tok={lg_res['flops']['gflops_per_tok']:.3f}"
                      if lg_res.get('flops') else "")
            print(f"\n  LG-only baseline: acc={lg_res['acc']:.4f}  "
                  f"prec={lg_res['prec']:.4f}  rec={lg_res['rec']:.4f}  "
                  f"f1={lg_res['f1']:.4f}  avg={lg_res['ms_avg']:.1f}ms{fl_str}")
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
                    lg_row["calib_risk"]     = "-"
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
                    if lg_res.get("flops") and "gflops_avg" in fieldnames:
                        lg_row["gflops_avg"]     = round(lg_res["flops"]["gflops_avg"], 3)
                        lg_row["gflops_per_tok"] = round(lg_res["flops"]["gflops_per_tok"], 4)
                        lg_row["lg_only_gflops_avg"]     = lg_row["gflops_avg"]
                        lg_row["lg_only_gflops_per_tok"] = lg_row["gflops_per_tok"]
                        lg_row["flops_ratio"]            = 1.0
                    writer.writerow(lg_row)
                writer.writerows(rows)
            print(f"\n  Summary saved → {out_path}")

    sce.unload(model)


if __name__ == "__main__":
    main()
