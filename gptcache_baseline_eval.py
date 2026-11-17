import argparse
import csv
import hashlib
import json
import os
import sys
import time

import faiss
import numpy as np
from tqdm import tqdm

_HERE = os.path.dirname(os.path.abspath(__file__))

_DATA_ROOT = os.environ.get("SAFECACHE_ROOT") or _HERE
sys.path.insert(0, _HERE)

_GPTCACHE_REPO = os.environ.get("GPTCACHE_REPO")
if _GPTCACHE_REPO and os.path.isdir(_GPTCACHE_REPO):
    sys.path.insert(0, _GPTCACHE_REPO)

from semantic_cache_eval import (
    DATASET_PATHS,
    load_data, dedup, stratified_split, set_seed,
    resolve_model_cfg, load_model, unload,
    run_lg_only, _metrics,
    make_flops_profile, faiss_search_flops, _flops_stats,
)
from jb_mutate import JB_DATASETS as _JB_DATASETS, load_jb_dataset

from gptcache.embedding import Onnx
from gptcache.manager import CacheBase, VectorBase, get_data_manager
from gptcache.manager.eviction import EvictionBase
from gptcache.similarity_evaluation import SearchDistanceEvaluation

STORE_ROOT     = os.path.join(_DATA_ROOT, "gptcache_store")
TOTAL_LOG_DIR  = os.path.join(_DATA_ROOT, "total_log")
EMB_CKPT_EVERY = 2000


def _prompts_digest(prompts):
    h = hashlib.sha1()
    for p in prompts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def embed_prompts(encoder, prompts, tag, store_dir):
    os.makedirs(store_dir, exist_ok=True)
    path = os.path.join(store_dir, f"emb_{tag}_{_prompts_digest(prompts)}.npy")
    n, dim = len(prompts), encoder.dimension
    if os.path.exists(path):
        embs = np.load(path)
        if embs.shape == (n, dim) and not np.isnan(embs[-1]).any():
            print(f"  [emb] {tag}: loaded {n:,} from {os.path.basename(path)}")
            return embs
        start = int(np.argmax(np.isnan(embs[:, 0]))) if np.isnan(embs[:, 0]).any() else n
    else:
        embs  = np.full((n, dim), np.nan, dtype=np.float32)
        start = 0
    for i in tqdm(range(start, n), initial=start, total=n,
                  desc=f"  emb {tag[:10]:<10}", ncols=72):
        embs[i] = encoder.to_embeddings(prompts[i])
        if (i + 1) % EMB_CKPT_EVERY == 0:
            np.save(path, embs)
    np.save(path, embs)
    return embs


def build_data_manager(store_dir, dim):
    os.makedirs(store_dir, exist_ok=True)
    sql_url = "sqlite:///" + os.path.join(store_dir, "sqlite.db")
    return get_data_manager(
        cache_base=CacheBase("sqlite", sql_url=sql_url),
        vector_base=VectorBase("faiss",
                               index_path=os.path.join(store_dir, "faiss.index"),
                               dimension=dim, top_k=1),
        eviction_base=EvictionBase("no_op_eviction"),
    )


_GPU_RES = None

def maybe_gpu_index(dm, force_cpu=False):
    global _GPU_RES
    if force_cpu or os.environ.get("JB_FAISS_CPU") == "1" or faiss.get_num_gpus() == 0:
        print("  [faiss] search on CPU")
        return
    if _GPU_RES is None:
        _GPU_RES = faiss.StandardGpuResources()
    co = faiss.GpuClonerOptions()
    co.useFloat16 = True
    dm.v._index = faiss.index_cpu_to_gpu(_GPU_RES, 0, dm.v._index, co)
    print(f"  [faiss] index on GPU (fp16 storage), "
          f"{dm.v._index.ntotal:,} vectors")


def build_cache(dm, index_prompts, index_labels, index_embs):
    n_have = dm.v._index.ntotal if hasattr(dm.v, "_index") else 0
    if n_have >= len(index_prompts):
        print(f"  [cache] already built ({n_have:,} vectors) — skipping insert")
        return
    if n_have > 0:
        raise RuntimeError(
            f"cache store is partially built ({n_have:,}/{len(index_prompts):,}); "
            "delete its directory and rerun")
    answers = ["harmful" if l == 1 else "harmless" for l in index_labels]
    B = 1000
    for i in tqdm(range(0, len(index_prompts), B), desc="  insert    ", ncols=72):
        dm.import_data(
            questions=index_prompts[i:i + B],
            answers=answers[i:i + B],
            embedding_datas=[e for e in index_embs[i:i + B]],
            session_ids=[None] * len(index_prompts[i:i + B]),
        )
    dm.flush()
    print(f"  [cache] inserted {len(index_prompts):,} entries")


def albert_embed_flops(encoder):
    from transformers import AutoConfig
    cfg  = AutoConfig.from_pretrained("GPTCache/paraphrase-albert-small-v2")
    T    = min(getattr(encoder.tokenizer, "model_max_length", 512), 512)
    d    = cfg.hidden_size
    d_ff = cfg.intermediate_size
    emb_proj = 2 * T * cfg.embedding_size * d
    attn     = 4 * (2 * T * d * d) + 4 * T * T * d
    ffn      = 2 * (2 * T * d * d_ff)
    return int(emb_proj + cfg.num_hidden_layers * (attn + ffn))


def lookup(dm, evaluator, encoder, prompt, emb=None):
    t0 = time.perf_counter()
    if emb is None:
        emb = encoder.to_embeddings(prompt)
    res = dm.search(emb, top_k=1)
    if not res:
        return -1.0, None, time.perf_counter() - t0
    top = res[0]
    cache_data = dm.get_scalar_data(top)
    score = evaluator.evaluation({}, {"search_result": top})
    t = time.perf_counter() - t0
    if cache_data is None:
        return -1.0, None, t
    ans = cache_data.answers[0].answer if hasattr(cache_data.answers[0], "answer") \
        else cache_data.answers[0]
    return float(score), (1 if ans == "harmful" else 0), t


def calibrate_threshold(cal_scores, cal_cached, cal_true, alpha, max_rank=4.0):
    s  = np.asarray(cal_scores)
    cl = np.asarray([c if c is not None else -1 for c in cal_cached])
    tv = np.asarray(cal_true)
    n_harmless = max(int((tv == 0).sum()), 1)
    n_harmful  = max(int((tv == 1).sum()), 1)

    cands = np.unique(np.percentile(s, np.linspace(0, 100, 500)))
    best = None
    for th in cands:
        hit   = s >= th
        fpr_h = float(((tv == 0) & hit & (cl == 1)).sum()) / n_harmless
        fpr_l = float(((tv == 1) & hit & (cl == 0)).sum()) / n_harmful
        if fpr_h <= alpha and fpr_l <= alpha:
            best = (float(th), fpr_h, fpr_l)
            break
    if best is None:
        best = (float(max_rank), 0.0, 0.0)
    th, fpr_h, fpr_l = best
    return th, th / max_rank, fpr_h, fpr_l


def guard_records(model_ref, cfg, test_ds, prompts, labels, limit, store_dir):
    gdir = os.path.join(STORE_ROOT, "guard_preds", cfg["key"])
    os.makedirs(gdir, exist_ok=True)
    sfx  = f"_limit{limit}" if limit else ""
    path = os.path.join(gdir, f"{test_ds}{sfx}.json")
    if os.path.exists(path):
        with open(path) as f:
            saved = json.load(f)
        if saved["n"] == len(prompts):
            if saved["records"] and "n_tok" in saved["records"][0]:
                print(f"  [guard] {test_ds}: loaded cached predictions ({saved['n']:,})")
                return saved["records"]
            print(f"  [guard] {test_ds}: cached predictions predate FLOPs "
                  f"accounting — re-measuring")
    if model_ref["model"] is None:
        print(f"\n  Loading guard model {cfg['key']} ...")
        model_ref["model"], model_ref["tok"] = load_model(cfg)
    flops_profile = make_flops_profile(model_ref["model"], cfg["arch"])
    res = run_lg_only(model_ref["model"], model_ref["tok"], cfg["arch"],
                      prompts, labels, flops_profile=flops_profile)
    records = res["_records"]
    with open(path, "w") as f:
        json.dump({"n": len(prompts), "model": cfg["key"],
                   "dataset": test_ds, "records": records}, f)
    return records


def compose(cache_rows, guard_recs, labels, theta, lookup_fl=0):
    preds, times, paths = [], [], []
    fl_list, tk_list    = [], []
    for row, g in zip(cache_rows, guard_recs):
        hit = row["cached"] is not None and row["score"] >= theta
        if hit:
            preds.append(row["cached"])
            times.append(row["t_cache"])
            paths.append("cache_harm" if row["cached"] == 1 else "cache_less")
            fl_list.append(lookup_fl)
        else:
            preds.append(g["pred"])
            times.append(row["t_cache"] + g["latency_ms"] / 1000.0)
            paths.append("lg")
            fl_list.append(lookup_fl + g.get("gflops", 0.0) * 1e9)
        tk_list.append(g.get("n_tok", 0))
    res = {}
    for path_name in ("cache_harm", "cache_less", "lg"):
        sel = [i for i, p in enumerate(paths) if p == path_name]
        res[path_name] = _metrics([labels[i] for i in sel],
                                  [preds[i] for i in sel],
                                  [times[i] for i in sel])
    res["combined"] = _metrics(list(labels), preds, times)
    res["flops"]    = _flops_stats(fl_list, tk_list)
    res["_preds"], res["_times"], res["_paths"] = preds, times, paths
    res["_flops_list"] = fl_list
    return res


def summary_row(mode, sim_th, theta, res, lg_res):
    com   = res["combined"]
    total = com["n"] if com["n"] else 1
    n_hit = res["cache_harm"]["n"] + res["cache_less"]["n"]
    row = {
        "mode":           mode,
        "sim_threshold":  round(sim_th, 4),
        "rank_threshold": round(theta, 4),
        "acc":  com["acc"], "prec": com["prec"], "rec": com["rec"], "f1": com["f1"],
        "ms_avg": com["ms_avg"], "ms_min": com["ms_min"], "ms_max": com["ms_max"],
        "hit_rate":       round(n_hit / total, 4),
        "cache_harm_n":   res["cache_harm"]["n"],
        "cache_harm_pct": round(res["cache_harm"]["n"] / total * 100, 1),
        "cache_harm_acc": res["cache_harm"]["acc"],
        "cache_less_n":   res["cache_less"]["n"],
        "cache_less_pct": round(res["cache_less"]["n"] / total * 100, 1),
        "cache_less_acc": res["cache_less"]["acc"],
        "lg_n":           res["lg"]["n"],
        "lg_pct":         round(res["lg"]["n"] / total * 100, 1),
        "lg_only_acc":  lg_res["acc"],  "lg_only_prec": lg_res["prec"],
        "lg_only_rec":  lg_res["rec"],  "lg_only_f1":   lg_res["f1"],
        "lg_only_ms_avg": lg_res["ms_avg"],
    }
    if res.get("flops"):
        row["gflops_avg"]     = round(res["flops"]["gflops_avg"], 3)
        row["gflops_per_tok"] = round(res["flops"]["gflops_per_tok"], 4)
        if lg_res.get("flops"):
            row["lg_only_gflops_avg"]     = round(lg_res["flops"]["gflops_avg"], 3)
            row["lg_only_gflops_per_tok"] = round(lg_res["flops"]["gflops_per_tok"], 4)
            if lg_res["flops"]["gflops_avg"] > 0:
                row["flops_ratio"] = round(
                    res["flops"]["gflops_avg"] / lg_res["flops"]["gflops_avg"], 4)
    return row


def lg_only_row(lg_res):
    total = lg_res["n"] if lg_res["n"] else 1
    row = {
        "mode": "lg_only", "sim_threshold": "-", "rank_threshold": "-",
        "acc": lg_res["acc"], "prec": lg_res["prec"],
        "rec": lg_res["rec"], "f1": lg_res["f1"],
        "ms_avg": lg_res["ms_avg"], "ms_min": lg_res["ms_min"],
        "ms_max": lg_res["ms_max"],
        "hit_rate": 0.0,
        "cache_harm_n": 0, "cache_harm_pct": 0.0, "cache_harm_acc": 0,
        "cache_less_n": 0, "cache_less_pct": 0.0, "cache_less_acc": 0,
        "lg_n": total, "lg_pct": 100.0,
        "lg_only_acc": lg_res["acc"], "lg_only_prec": lg_res["prec"],
        "lg_only_rec": lg_res["rec"], "lg_only_f1": lg_res["f1"],
        "lg_only_ms_avg": lg_res["ms_avg"],
    }
    if lg_res.get("flops"):
        row["gflops_avg"]             = round(lg_res["flops"]["gflops_avg"], 3)
        row["gflops_per_tok"]         = round(lg_res["flops"]["gflops_per_tok"], 4)
        row["lg_only_gflops_avg"]     = row["gflops_avg"]
        row["lg_only_gflops_per_tok"] = row["gflops_per_tok"]
        row["flops_ratio"]            = 1.0
    return row


def print_res(tag, res):
    com = res["combined"]
    n_hit = res["cache_harm"]["n"] + res["cache_less"]["n"]
    total = com["n"] if com["n"] else 1
    fl_str = (f" GF/ex={res['flops']['gflops_avg']:.1f}"
              f" GF/tok={res['flops']['gflops_per_tok']:.3f}"
              if res.get("flops") else "")
    print(f"  [{tag}] acc={com['acc']:.4f} prec={com['prec']:.4f} "
          f"rec={com['rec']:.4f} f1={com['f1']:.4f} "
          f"ms_avg={com['ms_avg']:.1f} hit={n_hit}/{total} ({n_hit/total*100:.1f}%)"
          f"{fl_str}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--cache-dataset", nargs="+", default=["wildguardmix", "aegis2", "toxicchat"],
                    choices=list(DATASET_PATHS.keys()))
    ap.add_argument("--test-dataset", nargs="+",
                    default=["wildguardmix", "toxicchat", "aegis2",
                             "simple", "openai", "harmbench"],
                    choices=list(DATASET_PATHS.keys()))
    ap.add_argument("--cache-source", choices=["cache", "train_full"], default="train_full",
                    help="'train_full' = the *_train_full files (default); "
                         "'cache' = the half-split *_cache files")
    ap.add_argument("--fpr-alpha", type=float, nargs="+", default=[],
                    help="optional: carve a 20%% calibration split and add "
                         "FPR-alpha-calibrated operating points (the baseline's "
                         "protocol). Off by default — GPTCache's paper uses a "
                         "static similarity_threshold, no calibration.")
    ap.add_argument("--sim-threshold", type=float, nargs="+",
                    default=[round(0.01 * i, 2) for i in range(101)],
                    help="GPTCache similarity_threshold sweep values, default "
                         "0.00..1.00 step 0.01 (0.8 = GPTCache default); "
                         "one summary row per value")
    ap.add_argument("--limit", type=int, default=None, help="cap test samples per dataset")
    ap.add_argument("--cache-limit", type=int, default=None,
                    help="cap cache samples per dataset (smoke tests only)")
    ap.add_argument("--summary-file", type=str, default=None)
    ap.add_argument("--per-example-log", action="store_true")
    ap.add_argument("--faiss-cpu", action="store_true",
                    help="keep the faiss index on CPU (stock GPTCache behavior); "
                         "default moves it to GPU fp16 like the baseline's "
                         "_to_gpu_index (JB_FAISS_CPU=1 also forces CPU)")
    ap.add_argument("--build-only", action="store_true",
                    help="build the cache store + calibrate thresholds, then exit "
                         "(CPU-only; no guard model / GPU needed)")
    args = ap.parse_args()

    set_seed()
    os.chdir(_DATA_ROOT)
    model_path = args.model
    if not os.path.exists(model_path) and model_path.startswith("./"):
        model_path = os.path.join(_DATA_ROOT, model_path[2:])
    cfg = resolve_model_cfg(model_path)

    use_calib = len(args.fpr_alpha) > 0
    store_tag = f"{args.cache_source}{'-cal' if use_calib else ''}_" \
                + "+".join(sorted(args.cache_dataset)) \
                + (f"_cl{args.cache_limit}" if args.cache_limit else "")
    store_dir = os.path.join(STORE_ROOT, store_tag)

    print("=" * 74)
    print("  GPTCache baseline  (Onnx albert + sqlite/faiss + SearchDistance)")
    print(f"  Model          : {model_path}")
    print(f"  Cache datasets : {args.cache_dataset}  (source={args.cache_source})")
    print(f"  Test  datasets : {args.test_dataset}")
    print(f"  fpr-alpha      : {args.fpr_alpha}   fixed sim_threshold: {args.sim_threshold}")
    print(f"  Store          : {store_dir}")
    print("=" * 74)

    index_prompts, index_labels = [], []
    calib_prompts, calib_labels = [], []
    for ds in args.cache_dataset:
        key  = "cache_data" if args.cache_source == "cache" else "train_full"
        path = DATASET_PATHS[ds][key]
        if path is None:
            raise SystemExit(f"dataset {ds} has no {key} split")
        tp, tl = load_data(os.path.join(_DATA_ROOT, path), ds, limit=args.cache_limit)
        tp, tl, n_rm = dedup(tp, tl)
        if use_calib:
            idx_mask, cal_mask = stratified_split(tl)
        else:
            idx_mask = np.ones(len(tl), dtype=bool)
            cal_mask = ~idx_mask
        print(f"  [{ds}] cache: {len(tp):,} after dedup (-{n_rm:,})  "
              f"index={int(idx_mask.sum()):,}  calib={int(cal_mask.sum()):,}  "
              f"harmful={int(tl.sum()):,}")
        index_prompts += [p for p, m in zip(tp, idx_mask) if m]
        index_labels  += list(tl[idx_mask])
        calib_prompts += [p for p, m in zip(tp, cal_mask) if m]
        calib_labels  += list(tl[cal_mask])
    index_labels = np.array(index_labels)
    calib_labels = np.array(calib_labels)

    print("\n  Loading GPTCache Onnx embedding (paraphrase-albert-onnx) ...")
    encoder = Onnx()
    if not hasattr(type(encoder.tokenizer), "encode_plus"):
        _tok = encoder.tokenizer
        _max = min(_tok.model_max_length, 512)
        encoder.tokenizer.encode_plus = (
            lambda text, padding="max_length", **kw: _tok(
                text, padding=padding, truncation=True, max_length=_max,
                return_token_type_ids=True, **kw))
    evaluator = SearchDistanceEvaluation()
    max_rank  = evaluator.range()[1]
    dm        = build_data_manager(store_dir, encoder.dimension)

    index_embs = embed_prompts(encoder, index_prompts, "index", store_dir)
    build_cache(dm, index_prompts, index_labels, index_embs)
    maybe_gpu_index(dm, force_cpu=args.faiss_cpu)

    emb_fl    = albert_embed_flops(encoder)
    search_fl = faiss_search_flops(dm.v._index)
    lookup_fl = emb_fl + search_fl
    print(f"  FLOPs/lookup   : albert {emb_fl/1e9:.2f} G + faiss "
          f"{search_fl/1e9:.3f} G = {lookup_fl/1e9:.2f} GFLOPs")

    operating_points = []
    if use_calib:
        calib_embs = embed_prompts(encoder, calib_prompts, "calib", store_dir)
        cal_scores, cal_cached = [], []
        for i in tqdm(range(len(calib_prompts)), desc="  calib     ", ncols=72):
            s, c, _ = lookup(dm, evaluator, encoder, calib_prompts[i],
                             emb=calib_embs[i])
            cal_scores.append(s)
            cal_cached.append(c)
        for a in args.fpr_alpha:
            th, sim, fh, fl = calibrate_threshold(cal_scores, cal_cached,
                                                  calib_labels, a, max_rank)
            print(f"  [calib] alpha={a}: rank_th={th:.4f} (sim_th={sim:.4f})  "
                  f"calib FPR harm={fh:.4f} less={fl:.4f}")
            operating_points.append((f"fpr{a}", sim, th))
    for st in args.sim_threshold:
        operating_points.append((f"fixed{st}", st, st * max_rank))

    if args.build_only:
        print("\n  --build-only: cache store built and thresholds calibrated. Done.")
        return

    model_ref = {"model": None, "tok": None}
    for td in args.test_dataset:
        print(f"\n{'─'*74}\n  Test dataset: {td}\n{'─'*74}")
        if td in _JB_DATASETS:
            tp, tl, _meta = load_jb_dataset(td, load_data, DATASET_PATHS,
                                            limit=args.limit)
        else:
            tp, tl = load_data(os.path.join(_DATA_ROOT, DATASET_PATHS[td]["test"]), td,
                               limit=args.limit)
        print(f"  {len(tp):,} samples (harmful={int(tl.sum()):,} "
              f"harmless={int((tl==0).sum()):,})")

        g_recs = guard_records(model_ref, cfg, td, tp, tl, args.limit, store_dir)
        lg_res = _metrics([r["true_label"] for r in g_recs],
                          [r["pred"] for r in g_recs],
                          [r["latency_ms"] / 1000.0 for r in g_recs])
        lg_res["flops"] = _flops_stats(
            [r.get("gflops", 0.0) * 1e9 for r in g_recs],
            [r.get("n_tok", 0) for r in g_recs])

        for p in tp[:3]:
            lookup(dm, evaluator, encoder, p)
        cache_rows = []
        for p in tqdm(tp, desc="  gptcache  ", ncols=72):
            s, c, t = lookup(dm, evaluator, encoder, p)
            cache_rows.append({"score": s, "cached": c, "t_cache": t})

        rows = [lg_only_row(lg_res)]
        for mode, sim, th in operating_points:
            res = compose(cache_rows, g_recs, tl, th, lookup_fl=lookup_fl)
            print_res(f"{td} {mode}", res)
            rows.append(summary_row(mode, sim, th, res, lg_res))

            if args.per_example_log and (mode.startswith("fpr") or mode == "fixed0.8"):
                os.makedirs(TOTAL_LOG_DIR, exist_ok=True)
                lp = os.path.join(TOTAL_LOG_DIR,
                                  f"gptcache_{cfg['label']}_{td}_{mode}.csv")
                with open(lp, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["idx", "true_label", "pred", "correct",
                                "latency_ms", "path", "score", "gflops", "prompt"])
                    for i, (p, l) in enumerate(zip(tp, tl)):
                        w.writerow([i, int(l), res["_preds"][i],
                                    int(res["_preds"][i] == l),
                                    round(res["_times"][i] * 1000, 2),
                                    res["_paths"][i],
                                    round(cache_rows[i]["score"], 4),
                                    round(res["_flops_list"][i] / 1e9, 4), p])

        print(f"  [lg-only] acc={lg_res['acc']:.4f} f1={lg_res['f1']:.4f} "
              f"ms_avg={lg_res['ms_avg']:.1f} "
              f"GF/ex={lg_res['flops']['gflops_avg']:.1f} "
              f"GF/tok={lg_res['flops']['gflops_per_tok']:.3f}")

        if args.summary_file:
            out = f"{args.summary_file}_{td}.csv"
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                for r in rows:
                    w.writerow({k: (round(v, 4) if isinstance(v, float) else v)
                                for k, v in r.items()})
            print(f"  summary → {out}")

    if model_ref["model"] is not None:
        unload(model_ref["model"])
    print("\nDone.")


if __name__ == "__main__":
    main()
