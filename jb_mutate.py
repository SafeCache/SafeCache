import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA_ROOT = os.environ.get("SAFECACHE_ROOT") or _HERE

MUTATIONS = ["roleplay", "encoding", "hypothetical_context",
             "auto_payload_splitting", "leetspeak", "translate", "append_text"]

DS_KEY = {
    "wildguardmix": "wildguardmix",
    "toxicchat":    "toxicchat",
    "aegis2":       "aegis2",
    "simple":       "simplesafety",
    "openai":       "oaimod",
    "harmbench":    "harmbench",
}
SRC_DATASETS = list(DS_KEY)

JB_ORIG   = "jb_orig"
JB_MIXED  = "jb_mixed"
VARIANTS  = ["orig", "mixed", "native"] + MUTATIONS
JB_MERGED = [JB_ORIG, JB_MIXED] + [f"jb_{m}" for m in MUTATIONS]

JB_SCOPED = {f"jb_{src}_{v}": (src, v)
             for src in DS_KEY for v in VARIANTS}
JB_DATASETS = JB_MERGED + list(JB_SCOPED)


def scoped_name(src, variant):
    return f"jb_{src}_{variant}"

JB_BENIGN_N = 1500
JB_SEED     = 20250803

VANILLA_ONLY = os.environ.get("JB_VANILLA_ONLY", "0") == "1"

NO_BENIGN = os.environ.get("JB_NO_BENIGN", "0") == "1"


def _already_adversarial(ds):
    out = set()
    if ds == "wildguardmix":
        with open(os.path.join(_DATA_ROOT, "data/wildguardmix/test.json")) as f:
            for line in f:
                r = json.loads(line)
                if r.get("adversarial"):
                    out.add(r["prompt"])
    elif ds == "toxicchat":
        import csv as _csv
        with open(os.path.join(_DATA_ROOT,
                               "data/toxic-chat/toxic-chat_test.csv")) as f:
            for r in _csv.DictReader(f):
                if str(r.get("jailbreaking", "0")) == "1":
                    out.add(r["user_input"].strip())
    return out


def _mutation_of(name):
    if name == JB_ORIG:
        return None
    if name == JB_MIXED:
        return "mixed"
    assert name.startswith("jb_"), name
    m = name[3:]
    if m not in MUTATIONS:
        raise ValueError(f"unknown mutation '{m}' (have {MUTATIONS})")
    return m


def parse_scoped(name):
    got = JB_SCOPED.get(name)
    if got is None:
        return None
    src, variant = got
    return src, (None if variant == "orig" else variant)


def has_native(src):
    return src in ("wildguardmix", "toxicchat")


_MAP_CACHE = {}

def mutation_maps(ds):
    if ds not in _MAP_CACHE:
        path = os.path.join(_DATA_ROOT, "data", "jailbreak_dataset",
                            f"{DS_KEY[ds]}_mutated_harmful_merged.json")
        with open(path) as f:
            rows = json.load(f)
        _MAP_CACHE[ds] = {r["original_prompt"]: r["mutations"] for r in rows}
    return _MAP_CACHE[ds]


def pick(muts, mutation, rng):
    if mutation == "mixed":
        avail = [m for m in MUTATIONS if muts.get(m)]
        if not avail:
            return None
        mutation = avail[rng.integers(len(avail))]
    text = muts.get(mutation)
    return (text, mutation) if text else None


def load_jb_dataset(name, load_data, dataset_paths, sources=None, limit=None):
    scoped = parse_scoped(name)
    if scoped is not None:
        return _load_scoped(scoped[0], scoped[1], load_data, dataset_paths,
                            limit=limit)
    mutation = _mutation_of(name)
    sources  = sources or SRC_DATASETS
    rng      = np.random.default_rng(JB_SEED)

    harm_p, harm_meta = [], []
    benign_by_src = {}
    for ds in sources:
        tp, tl = load_data(os.path.join(_DATA_ROOT, dataset_paths[ds]["test"]),
                           ds, limit=limit)
        maps = mutation_maps(ds)
        skip = _already_adversarial(ds) if VANILLA_ONLY else set()
        benign_by_src[ds] = [p for p, l in zip(tp, tl) if l == 0 and p not in skip]
        for p, l in zip(tp, tl):
            if l != 1 or p not in maps or p in skip:
                continue
            if mutation is None:
                harm_p.append(p)
                harm_meta.append(dict(src=ds, mutation="orig"))
                continue
            got = pick(maps[p], mutation, rng)
            if got is None:
                continue
            text, used = got
            harm_p.append(text)
            harm_meta.append(dict(src=ds, mutation=used))

    b_rng  = np.random.default_rng(JB_SEED + 1)
    total  = sum(len(v) for v in benign_by_src.values())
    ben_p, ben_meta = [], []
    if NO_BENIGN:
        benign_by_src = {}
    for ds, pool in benign_by_src.items():
        if not pool:
            continue
        n = min(len(pool), int(round(JB_BENIGN_N * len(pool) / total)))
        for i in b_rng.choice(len(pool), n, replace=False):
            ben_p.append(pool[i])
            ben_meta.append(dict(src=ds, mutation=""))

    prompts = harm_p + ben_p
    labels  = np.array([1] * len(harm_p) + [0] * len(ben_p))
    meta    = harm_meta + ben_meta
    return prompts, labels, meta


def _load_scoped(src, mutation, load_data, dataset_paths, limit=None):
    rng  = np.random.default_rng(JB_SEED)
    tp, tl = load_data(os.path.join(_DATA_ROOT, dataset_paths[src]["test"]),
                       src, limit=limit)
    maps = mutation_maps(src)
    skip = _already_adversarial(src) if VANILLA_ONLY else set()

    if mutation == "native":
        adv = _already_adversarial(src)
        prompts = [p for p, l in zip(tp, tl) if l == 1 and p in adv]
        meta = [dict(src=src, mutation="native") for _ in prompts]
        if not NO_BENIGN:
            ben = [p for p, l in zip(tp, tl) if l == 0 and p in adv]
            prompts += ben
            meta += [dict(src=src, mutation="") for _ in ben]
        labels = np.array([1] * (len(prompts) - (0 if NO_BENIGN else len(ben)))
                          + ([0] * len(ben) if not NO_BENIGN else []))
        return prompts, labels, meta

    prompts, labels, meta = [], [], []
    for p, l in zip(tp, tl):
        if l == 0:
            if NO_BENIGN or p in skip:
                continue
            prompts.append(p); labels.append(0)
            meta.append(dict(src=src, mutation=""))
            continue
        if p not in maps or p in skip:
            continue
        if mutation is None:
            prompts.append(p); labels.append(1)
            meta.append(dict(src=src, mutation="orig"))
            continue
        got = pick(maps[p], mutation, rng)
        if got is None:
            continue
        text, used = got
        prompts.append(text); labels.append(1)
        meta.append(dict(src=src, mutation=used))
    return prompts, np.array(labels), meta


def mutate_stream(prompts, labels, srcs, mutation, ratio=1.0, seed=JB_SEED):
    rng   = np.random.default_rng(seed)
    flags = [""] * len(prompts)
    out   = list(prompts)

    by_src = {}
    for i, (l, s) in enumerate(zip(labels, srcs)):
        if l == 1 and s in DS_KEY:
            by_src.setdefault(s, []).append(i)

    for ds, idxs in by_src.items():
        maps = mutation_maps(ds)
        skip = _already_adversarial(ds) if VANILLA_ONLY else set()
        elig = [i for i in idxs if out[i] in maps and out[i] not in skip]
        n    = int(round(len(elig) * ratio))
        if n <= 0:
            continue
        for i in rng.choice(elig, n, replace=False):
            got = pick(maps[out[i]], mutation, rng)
            if got is None:
                continue
            out[i], flags[i] = got[0], got[1]
    return out, flags
