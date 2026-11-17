import argparse, csv, gc, json, os, random, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
    Llama4ForConditionalGeneration,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
from tqdm import tqdm

GUARD_PATH    = "./models/Llama-Guard-3-8B"

_AEGIS_HEADER = """\
[INST] Task: Check if there is unsafe content in 'User' messages in conversations according to our safety policy with the below categories.

<BEGIN UNSAFE CONTENT CATEGORIES>
O1: PII/Privacy.
O2: Violence.
O3: Suicide and Self Harm.
O4: Criminal Planning/Confessions.
O5: Hate/Identity Hate.
O6: Controlled/Regulated Substances.
O7: Guns and Illegal Weapons.
O8: Profanity.
O9: Sexual.
O10: Harassment.
O11: Sexual (minor).
O12: Threat.
O13: Needs Caution.
<END UNSAFE CONTENT CATEGORIES>

<BEGIN CONVERSATION>
User: """

_AEGIS_FOOTER = """\

<END CONVERSATION>

Provide your safety assessment for the above conversation:
- First line must read 'safe' or 'unsafe'.
- If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""


def build_aegis_prompt(prompt_text: str) -> str:
    return _AEGIS_HEADER + prompt_text + _AEGIS_FOOTER


def _get_aegis_base_adapter(model_path):
    import re
    dirname = os.path.basename(os.path.normpath(model_path))
    if re.fullmatch(r'Aegis-LlamaGuard-\w+-[\d.]+', dirname):
        return None
    m = re.match(r'(Aegis-LlamaGuard-\w+-[\d.]+)-', dirname)
    if m:
        aegis_name = m.group(1)
        models_dir = os.path.dirname(os.path.abspath(model_path))
        candidate  = os.path.join(models_dir, aegis_name)
        if os.path.isdir(candidate):
            return candidate
    return None


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
LAYER_INDICES = [5, 10, 15, 20, 25, 30]
D_MODEL       = 4096
PROJ_HIDDEN   = 1024
PROJ_DIM      = 256
LAMBDA1       = 1.0
LAMBDA2       = 0.5
ALIGN_WEIGHT  = 1.0
TEMPERATURE   = 0.07
KL_TEMP       = 1.0
BATCH_SIZE    = 8
GRAD_ACCUM    = 4
LEARNING_RATE = 1e-3
KL_LR         = 2e-4
WARMUP_RATIO  = 0.1
KL_EPOCHS     = 5
MAX_SEQ_LEN   = 1024


def detect_arch(model_path):
    p = model_path.lower()
    if "guard-4" in p:
        return "lg4"
    if "shieldgemma" in p:
        return "shieldgemma"
    if "aegis-llamaguard" in p:
        return "aegis"
    return "causal"

GPU0_RESERVE_GIB = 6

def _balanced_max_memory(reserve_gib=GPU0_RESERVE_GIB):
    n = torch.cuda.device_count()
    if n <= 1:
        return None
    max_mem = {}
    for i in range(n):
        total_gib = torch.cuda.get_device_properties(i).total_memory / 1024**3
        budget = total_gib - reserve_gib if i == 0 else total_gib * 0.95
        max_mem[i] = f"{max(1, int(budget))}GiB"
    return max_mem


def load_backbone(model_path):
    arch = detect_arch(model_path)
    max_memory = _balanced_max_memory()
    adapter_cfg = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        import json as _json
        base_path = _json.load(open(adapter_cfg))["base_model_name_or_path"]
    else:
        base_path = model_path

    if arch == "lg4":
        tok   = AutoProcessor.from_pretrained(base_path)
        model = Llama4ForConditionalGeneration.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory,
        )
        if getattr(model.config.text_config, "attention_chunk_size", None) is None:
            model.config.text_config.attention_chunk_size = 8192
    elif arch == "aegis":
        if not os.path.exists(base_path):
            base_path = "./models/LlamaGuard-7b"
        tok   = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory,
        )
        aegis_base = _get_aegis_base_adapter(model_path)
        if aegis_base:
            print(f"  Merging original Aegis adapter: {aegis_base}")
            model = PeftModel.from_pretrained(model, aegis_base)
            model = model.merge_and_unload()
    else:
        tok   = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto",
            max_memory=max_memory,
        )

    if os.path.exists(adapter_cfg):
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()

    if arch in ("causal", "shieldgemma", "aegis") and tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    return model, tok, arch

def get_d_model(model, arch):
    return getattr(model.config, "text_config", model.config).hidden_size

_TYPED_CONTENT_CACHE = {}

def needs_typed_content(tok):
    key = id(tok)
    if key not in _TYPED_CONTENT_CACHE:
        probe = "XCONTENTPROBEX"
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": probe}],
            add_generation_prompt=True, tokenize=False,
        )
        _TYPED_CONTENT_CACHE[key] = probe not in rendered
    return _TYPED_CONTENT_CACHE[key]

def make_input_ids(tok, arch, prompt):
    if arch == "lg4":
        chat = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        out  = tok.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        return out["input_ids"][0].tolist()
    elif arch == "shieldgemma":
        chat = [{"role": "user", "content": prompt}]
        return tok.apply_chat_template(
            chat, guideline=DEFAULT_GUIDELINE,
            add_generation_prompt=True, tokenize=True, return_dict=False,
        )
    elif arch == "aegis":
        return list(tok.encode(build_aegis_prompt(prompt), add_special_tokens=True))
    else:
        content = [{"type": "text", "text": prompt}] if needs_typed_content(tok) else prompt
        chat = [{"role": "user", "content": content}]
        return tok.apply_chat_template(chat, add_generation_prompt=True, tokenize=True,
                                        return_dict=False)

def get_safe_unsafe_ids(tok, arch):
    t = tok.tokenizer if arch == "lg4" else tok
    pair = ("No", "Yes") if arch == "shieldgemma" else ("safe", "unsafe")
    ids  = [t.encode(w, add_special_tokens=False) for w in pair]
    assert all(len(i) == 1 for i in ids), \
        f"label words must be single tokens, got {pair} -> {ids}"
    return ids[0][0], ids[1][0]

def get_eos_id(tok, arch):
    t = tok.tokenizer if arch == "lg4" else tok
    return t.eos_token_id


def _wg_label(d):
    if d.get("prompt_harm_label") is None: return None
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
        "train":       "data/wildguardmix/train.json",
        "train_full":  "data/wildguardmix/train_full.json",
        "test":        "data/wildguardmix/test.json",
        "label_fn":    _wg_label,
        "prompt_field": "prompt",
    },
    "beavertails": {
        "train":       "data/beavertails/beavertails_train.jsonl",
        "train_full":  None,
        "test":        "data/beavertails/beavertails_test.jsonl",
        "label_fn":    _bt_label,
        "prompt_field": "prompt",
    },
    "lmsys": {
        "train":       "data/lmsys-chat-1m/train.json",
        "train_full":  None,
        "test":        "data/lmsys-chat-1m/test.json",
        "label_fn":    _lmsys_label,
        "prompt_field": "prompt",
    },
    "toxicchat": {
        "train":       "data/toxic-chat/toxic-chat_train.csv",
        "train_full":  "data/toxic-chat/toxic-chat_train_full.csv",
        "test":        "data/toxic-chat/toxic-chat_test.csv",
        "label_fn":    _tc_label,
        "prompt_field": "user_input",
    },
    "aegis2": {
        "train":       "data/aegis2/aegis2_train.csv",
        "train_full":  "data/aegis2/aegis2_train_full.csv",
        "test":        "data/aegis2/aegis2_test.csv",
        "label_fn":    _aegis2_label,
        "prompt_field": "prompt",
    },
}


def load_data(path, label_fn, limit=None, prompt_field="prompt"):
    records = []
    is_csv  = path.endswith(".csv")
    with open(path, "r", newline="" if is_csv else None) as f:
        rows = csv.DictReader(f) if is_csv else (json.loads(l) for l in f)
        for d in rows:
            label = label_fn(d)
            if label is None: continue
            prompt = d.get(prompt_field, "").strip()
            if not prompt: continue
            records.append({"prompt": prompt, "label": label})
            if limit and len(records) >= limit: break
    return records


def dedup(records):
    seen, out = set(), []
    for r in records:
        if r["prompt"] not in seen:
            seen.add(r["prompt"])
            out.append(r)
    removed = len(records) - len(out)
    if removed:
        print(f"  Dedup: removed {removed:,} duplicates ({len(out):,} unique remain)")
    return out


class GuardDataset(Dataset):
    def __init__(self, records, tok, arch="causal", max_len=MAX_SEQ_LEN):
        self.items = []
        skipped = 0
        t = tok.tokenizer if arch == "lg4" else tok
        eos_id = get_eos_id(tok, arch)
        for r in records:
            input_ids = make_input_ids(tok, arch, r["prompt"])
            label_str = SG_LABEL[r["label"]] if arch == "shieldgemma" else r["label"]
            label_ids = t.encode(label_str + "\n", add_special_tokens=False)
            label_ids += [eos_id]

            full_ids  = input_ids + label_ids
            lm_labels = [-100] * len(input_ids) + label_ids

            if len(full_ids) > max_len:
                keep = max_len - len(label_ids)
                if keep < 20: skipped += 1; continue
                full_ids  = input_ids[:keep] + label_ids
                lm_labels = [-100] * keep + label_ids
                input_ids = input_ids[:keep]

            self.items.append({
                "input_ids": torch.tensor(full_ids,  dtype=torch.long),
                "lm_labels": torch.tensor(lm_labels, dtype=torch.long),
                "cls_label": torch.tensor(0 if r["label"] == "safe" else 1, dtype=torch.long),
                "input_len": len(input_ids),
            })

        self._arch = arch
        print(f"  Dataset: {len(self.items):,} samples  "
              f"({skipped:,} skipped — exceeded {max_len} tokens)")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx): return self.items[idx]


def collate_fn(batch):
    max_len    = max(x["input_ids"].shape[0] for x in batch)
    input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long)
    lm_labels  = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long)
    cls_labels = torch.stack([x["cls_label"] for x in batch])
    input_lens = torch.tensor([x["input_len"] for x in batch], dtype=torch.long)
    for i, x in enumerate(batch):
        n = x["input_ids"].shape[0]
        input_ids[i, :n] = x["input_ids"]
        lm_labels[i, :n] = x["lm_labels"]
        attn_mask[i, :n] = 1
    return {
        "input_ids":  input_ids,
        "lm_labels":  lm_labels,
        "attn_mask":  attn_mask,
        "cls_labels": cls_labels,
        "input_lens": input_lens,
    }


class ProjectionHead(nn.Module):
    def __init__(self, d_model=D_MODEL, hidden=PROJ_HIDDEN, out_dim=PROJ_DIM):
        super().__init__()
        self.fc1        = nn.Linear(d_model, hidden)
        self.act        = nn.GELU()
        self.fc2        = nn.Linear(hidden, out_dim)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x):
        h    = self.act(self.fc1(x))
        emb  = self.fc2(h)
        emb  = F.normalize(emb, p=2, dim=-1)
        logits = self.classifier(emb)
        return emb, logits


class MultiHeadProjector(nn.Module):
    def __init__(self, layer_indices=LAYER_INDICES, d_model=D_MODEL,
                 hidden=PROJ_HIDDEN, out_dim=PROJ_DIM):
        super().__init__()
        self.heads = nn.ModuleDict({
            str(li): ProjectionHead(d_model, hidden, out_dim) for li in layer_indices
        })

    def forward(self, hidden_by_layer):
        return {
            int(li): self.heads[str(li)](h)
            for li, h in hidden_by_layer.items()
        }


class SupConLoss(nn.Module):
    def __init__(self, temperature=TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        N      = features.shape[0]
        device = features.device

        if N < 2:
            return (features * 0).sum()

        sim = torch.matmul(features, features.T) / self.temperature

        labels  = labels.view(-1, 1)
        pos_mask = torch.eq(labels, labels.T).float().to(device)
        eye      = torch.eye(N, device=device)
        pos_mask = pos_mask * (1 - eye)

        if pos_mask.sum() == 0:
            return (features * 0).sum()

        neg_mask   = 1 - eye
        exp_sim    = torch.exp(sim) * neg_mask
        log_prob   = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-8)

        n_pos      = pos_mask.sum(1).clamp(min=1)
        loss       = -(pos_mask * log_prob).sum(1) / n_pos

        return loss.mean()


def label_logits_from_hidden(model, h_last):
    head = model.get_output_embeddings()
    h    = h_last.to(device=head.weight.device, dtype=head.weight.dtype)
    logits = head(h)
    cap = getattr(getattr(model.config, "text_config", model.config),
                  "final_logit_softcapping", None)
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    return logits


def kl_consistency_loss(head_logits, lg3_logits_2class, temperature=KL_TEMP):
    log_p_s = F.log_softmax(head_logits       / temperature, dim=-1)
    log_p_t = F.log_softmax(lg3_logits_2class / temperature, dim=-1)
    p_s     = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum(dim=-1).mean()
    if temperature != 1.0:
        kl = kl * temperature ** 2
    return kl


def train(model, tok, dataset, projector, epochs, grad_accum, output_dir,
          loss_mode="combined", oversample=False, lr=LEARNING_RATE,
          align_weight=0.0):
    from torch.utils.data import WeightedRandomSampler
    from transformers import get_linear_schedule_with_warmup

    device    = next(model.parameters()).device

    if oversample:
        labels     = [item["cls_label"].item() for item in dataset]
        n_safe     = labels.count(0)
        n_unsafe   = labels.count(1)
        class_w    = {0: 1.0 / n_safe, 1: 1.0 / n_unsafe}
        sample_w   = [class_w[l] for l in labels]
        n_majority = max(n_safe, n_unsafe)
        sampler    = WeightedRandomSampler(sample_w, num_samples=2 * n_majority, replacement=True)
        loader     = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler,
                                collate_fn=collate_fn, num_workers=0)
        print(f"  oversample     : ON  (safe={n_safe:,}  unsafe={n_unsafe:,}  "
              f"samples/epoch={2*n_majority:,}  w_safe={class_w[0]:.5f}  w_unsafe={class_w[1]:.5f})")
    else:
        loader     = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                collate_fn=collate_fn, num_workers=0)

    optimizer = torch.optim.AdamW(projector.parameters(), lr=lr)
    supcon    = SupConLoss()

    kl_weight = LAMBDA2 if loss_mode == "combined" else 1.0

    arch = getattr(dataset, '_arch', 'causal') if hasattr(dataset, '_arch') else 'causal'
    safe_id, unsafe_id = get_safe_unsafe_ids(tok, arch)

    use_supcon = loss_mode in ("combined", "supcon")
    use_kl     = loss_mode in ("combined", "kl")
    use_align  = align_weight > 0
    last_li    = max(LAYER_INDICES)
    print(f"  head input     : last instruction token")
    if use_align:
        print(f"  align          : λ3={align_weight}  (heads < L{last_li} pulled "
              f"toward L{last_li} embedding, teacher detached)")

    total_steps  = max(1, (len(loader) * epochs + grad_accum - 1) // grad_accum)
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"  loss mode      : {loss_mode}")
    print(f"  total steps    : {total_steps:,}  (warmup {warmup_steps:,})")
    print(f"  effective batch: {BATCH_SIZE * grad_accum}")
    print(f"  lr             : {lr}")
    if loss_mode == "combined":
        print(f"  λ1={LAMBDA1}  λ2={LAMBDA2}  T_supcon={TEMPERATURE}  T_kl={KL_TEMP}")
    elif loss_mode == "supcon":
        print(f"  λ1={LAMBDA1}  T_supcon={TEMPERATURE}")
    else:
        print(f"  pure KL (FastBERT Eq. 6, unweighted sum)  T_kl={KL_TEMP}")

    projector.train()
    optimizer.zero_grad()
    t0 = time.perf_counter()

    for epoch in range(epochs):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=72)

        for step, batch in enumerate(pbar):
            input_ids  = batch["input_ids"].to(device)
            attn_mask  = batch["attn_mask"].to(device)
            cls_labels = batch["cls_labels"].to(device)
            input_lens = batch["input_lens"]

            with torch.no_grad():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    output_hidden_states=True,
                    logits_to_keep=1,
                )
                hidden_by_layer = {}
                pos = (input_lens - 1).clamp(min=0)
                for li in LAYER_INDICES:
                    hs = out.hidden_states[li + 1]
                    idx = pos.to(hs.device).view(-1, 1, 1).expand(-1, 1, hs.shape[-1])
                    hidden_by_layer[li] = hs.gather(1, idx).squeeze(1).float().to(device)

                if use_kl:
                    hs_final = out.hidden_states[-1]
                    idx      = pos.to(hs_final.device).view(-1, 1, 1).expand(-1, 1, hs_final.shape[-1])
                    h_last   = hs_final.gather(1, idx).squeeze(1)
                    logits   = label_logits_from_hidden(model, h_last)
                    lg3_2class = torch.stack(
                        [logits[:, safe_id], logits[:, unsafe_id]], dim=1
                    ).float().to(device)
                else:
                    lg3_2class = None

            del out

            head_outputs = projector(hidden_by_layer)

            teacher = head_outputs[last_li][0].detach() if use_align else None

            total_loss = torch.tensor(0.0, device=device)
            for li in LAYER_INDICES:
                emb, head_logits = head_outputs[li]
                if use_supcon:
                    total_loss = total_loss + LAMBDA1 * supcon(emb, cls_labels)
                if use_kl:
                    total_loss = total_loss + kl_weight * kl_consistency_loss(head_logits, lg3_2class)
                if use_align and li != last_li:
                    align = (1.0 - (emb * teacher).sum(dim=-1)).mean()
                    total_loss = total_loss + align_weight * align

            loss = total_loss / grad_accum
            loss.backward()
            epoch_loss += total_loss.item()

            if (step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=f"{epoch_loss/(step+1):.4f}")

        print(f"  Epoch {epoch+1} avg loss: {epoch_loss/len(loader):.4f}")

        if epoch < epochs - 1:
            save_heads(projector, os.path.join(output_dir, f"epoch_{epoch+1}"),
                       loss_mode, align_weight)

    elapsed = time.perf_counter() - t0
    print(f"  Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    save_heads(projector, output_dir, loss_mode, align_weight)
    return elapsed


def save_heads(projector, output_dir, loss_mode="combined", align_weight=0.0):
    os.makedirs(output_dir, exist_ok=True)
    for li in LAYER_INDICES:
        path = os.path.join(output_dir, f"head_{li}.pt")
        torch.save(projector.heads[str(li)].state_dict(), path)
    first_head = next(iter(projector.heads.values()))
    actual_d_model = first_head.fc1.weight.shape[1]
    actual_hidden  = first_head.fc1.weight.shape[0]
    actual_out_dim = first_head.fc2.weight.shape[0]
    cfg = {"layer_indices": LAYER_INDICES, "d_model": actual_d_model,
           "proj_hidden": actual_hidden, "proj_dim": actual_out_dim,
           "loss_mode": loss_mode}
    if align_weight:
        cfg["align_weight"] = align_weight
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved heads → {output_dir}/head_{{layer}}.pt")


def load_heads(output_dir, device):
    with open(os.path.join(output_dir, "config.json")) as f:
        cfg = json.load(f)
    first_ckpt = torch.load(
        os.path.join(output_dir, f"head_{cfg['layer_indices'][0]}.pt"),
        map_location="cpu",
    )
    actual_d_model = first_ckpt["fc1.weight"].shape[1]
    actual_hidden  = first_ckpt["fc1.weight"].shape[0]
    actual_out_dim = first_ckpt["fc2.weight"].shape[0]
    projector = MultiHeadProjector(cfg["layer_indices"], actual_d_model,
                                   actual_hidden, actual_out_dim).to(device)
    projector.heads[str(cfg["layer_indices"][0])].load_state_dict(first_ckpt)
    for li in cfg["layer_indices"][1:]:
        path = os.path.join(output_dir, f"head_{li}.pt")
        projector.heads[str(li)].load_state_dict(
            torch.load(path, map_location=device)
        )
    projector.pool_cfg = cfg.get("pool")
    projector.eval()
    return projector


def evaluate(model, tok, records, projector, label_fn, arch="causal"):
    device    = next(model.parameters()).device
    safe_id, unsafe_id = get_safe_unsafe_ids(tok, arch)

    all_preds  = {li: [] for li in LAYER_INDICES}
    lg_preds   = []
    true_labels = []

    projector.eval()
    for r in tqdm(records, desc="  eval", ncols=72):
        ids = make_input_ids(tok, arch, r["prompt"])
        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
        if input_ids.shape[1] > MAX_SEQ_LEN:
            continue

        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True,
                        logits_to_keep=1)
            hidden_by_layer = {}
            for li in LAYER_INDICES:
                hs = out.hidden_states[li + 1]
                hidden_by_layer[li] = hs[:, -1, :].float().to(device)
            lg_logit = out.logits[0, -1, :]
            lg_pred  = 0 if lg_logit[safe_id] >= lg_logit[unsafe_id] else 1
            lg_preds.append(lg_pred)

            head_out = projector(hidden_by_layer)
            for li in LAYER_INDICES:
                _, logits = head_out[li]
                pred = logits.argmax(dim=-1).item()
                all_preds[li].append(pred)

        true_labels.append(0 if r["label"] == "safe" else 1)

    print(f"\n{'='*68}")
    print(f"  Evaluation Results  ({len(true_labels):,} samples)")
    print(f"{'='*68}")
    print(f"  {'':12} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}")
    print(f"  {'─'*48}")

    for li in LAYER_INDICES:
        preds = all_preds[li]
        acc   = accuracy_score(true_labels,  preds)
        prec  = precision_score(true_labels, preds, zero_division=0)
        rec   = recall_score(true_labels,    preds, zero_division=0)
        f1    = f1_score(true_labels,        preds, zero_division=0)
        print(f"  Head layer {li:>2}   {acc:7.4f} {prec:7.4f} {rec:7.4f} {f1:7.4f}")

    acc   = accuracy_score(true_labels,  lg_preds)
    prec  = precision_score(true_labels, lg_preds, zero_division=0)
    rec   = recall_score(true_labels,    lg_preds, zero_division=0)
    f1    = f1_score(true_labels,        lg_preds, zero_division=0)
    print(f"  {'─'*48}")
    print(f"  {'LG backbone':12} {acc:7.4f} {prec:7.4f} {rec:7.4f} {f1:7.4f}")
    print(f"{'='*68}")


def main():
    global LAYER_INDICES
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    type=str, nargs="+", required=True,
                        choices=list(DATASET_REGISTRY.keys()),
                        help="One or more datasets to train on (combined)")
    parser.add_argument("--model",      type=str, default=GUARD_PATH,
                        help="Path to backbone model (default: base LG3). "
                             "Pass a fine-tuned path to train heads on top of it.")
    parser.add_argument("--loss",       type=str, default="combined",
                        choices=["combined", "supcon", "kl"],
                        help="Loss function: combined (SupCon+KL), supcon, or kl")
    parser.add_argument("--epochs",     type=int,  default=None,
                        help="Default: 5 for --loss kl (FastBERT §4.3), else 4")
    parser.add_argument("--lr",         type=float, default=None,
                        help="Default: 2e-4 for --loss kl (FastBERT §4.3), else 1e-3")
    parser.add_argument("--limit",      type=int,  default=None)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--oversample", action="store_true",
                        help="Oversample minority class so each batch sees balanced labels (useful for imbalanced datasets like toxicchat)")
    parser.add_argument("--full", action="store_true",
                        help="Use train_full split instead of default train split")
    parser.add_argument("--proj-dim", type=int, default=PROJ_DIM,
                        help=f"Head output embedding dimension (default {PROJ_DIM}). "
                             "Small dims (16-64) act as an information bottleneck "
                             "that strips label-irrelevant topic directions and "
                             "buy a smaller/faster FAISS index.")
    parser.add_argument("--align-weight", type=float, default=0.0,
                        help="λ3 for cross-layer embedding alignment: pull each "
                             "non-final head's embedding toward the final head's "
                             "(teacher detached). 0 disables (default); try "
                             f"{ALIGN_WEIGHT} to shift separability to earlier "
                             "layers. Composes with any --loss mode.")
    parser.add_argument("--output_dir", type=str,  default=None)
    args = parser.parse_args()

    set_seed()
    guard_path   = args.model
    arch         = detect_arch(guard_path)
    datasets     = args.dataset
    dataset_tag  = "+".join(datasets)
    full_tag     = "-full" if args.full else ""
    output_dir   = args.output_dir or f"models/multihead-{dataset_tag}-{args.loss}{full_tag}"

    loss_mode = args.loss

    epochs = args.epochs if args.epochs is not None else (KL_EPOCHS if loss_mode == "kl" else 1)
    lr     = args.lr     if args.lr     is not None else (KL_LR     if loss_mode == "kl" else LEARNING_RATE)

    loss_labels = {
        "combined": f"λ1={LAMBDA1} * SupCon + λ2={LAMBDA2} * KL",
        "supcon":   f"λ1={LAMBDA1} * SupCon only",
        "kl":       "KL only (FastBERT Eq. 5–6 self-distillation)",
    }
    print("\nLoading backbone (frozen) ...")
    model, tok, arch = load_backbone(guard_path)
    d_model = get_d_model(model, arch)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    device = next(model.parameters()).device

    n_layers      = getattr(getattr(model.config, "text_config", model.config), "num_hidden_layers", 32)
    LAYER_INDICES = list(range(n_layers))

    print("=" * 68)
    print(f"  Multi-Head Projection Training  ({dataset_tag})")
    print(f"  Backbone : {guard_path}  [arch={arch}]")
    print(f"  Layers   : {LAYER_INDICES}")
    print(f"  Loss     : {loss_labels[loss_mode]}  (per head, summed)")
    if args.align_weight > 0:
        print(f"  Align    : + λ3={args.align_weight} * (1 - cos(emb_i, emb_last)) for i < last")
    print(f"  Head in  : last instruction token")
    print(f"  Output   : {output_dir}")
    print("=" * 68)
    print(f"  Backbone frozen — arch={arch}  d_model={d_model}")

    projector = MultiHeadProjector(LAYER_INDICES, d_model,
                                   PROJ_HIDDEN, args.proj_dim).to(device)
    n_params  = sum(p.numel() for p in projector.parameters())
    print(f"  Projection heads — {n_params:,} trainable params  "
          f"(Linear({d_model},{PROJ_HIDDEN})->GELU->Linear({PROJ_HIDDEN},{PROJ_DIM})->L2Norm)")

    t_train = 0.0
    if not args.eval_only:
        train_records = []
        for ds_name in datasets:
            cfg = DATASET_REGISTRY[ds_name]
            if args.full:
                train_path = cfg.get("train_full") or cfg["train"]
                if cfg.get("train_full") is None:
                    print(f"  [warn] No train_full for '{ds_name}', falling back to default train split")
            else:
                train_path = cfg["train"]
            print(f"\nLoading train data from {train_path} ...")
            records = load_data(train_path, cfg["label_fn"], limit=args.limit,
                                prompt_field=cfg.get("prompt_field", "prompt"))
            records = dedup(records)
            n_safe   = sum(1 for r in records if r["label"] == "safe")
            n_unsafe = sum(1 for r in records if r["label"] == "unsafe")
            print(f"  {len(records):,} records  (safe={n_safe:,}  unsafe={n_unsafe:,})")
            train_records.extend(records)

        if len(datasets) > 1:
            train_records = dedup(train_records)
            n_safe   = sum(1 for r in train_records if r["label"] == "safe")
            n_unsafe = sum(1 for r in train_records if r["label"] == "unsafe")
            print(f"\n  Combined (after dedup): {len(train_records):,}  "
                  f"(safe={n_safe:,}  unsafe={n_unsafe:,})")

        dataset  = GuardDataset(train_records, tok, arch=arch)

        print(f"\nTraining projection heads ...")
        t_train = train(model, tok, dataset, projector, epochs, GRAD_ACCUM, output_dir,
                        loss_mode=loss_mode, oversample=args.oversample, lr=lr,
                        align_weight=args.align_weight)
    else:
        print(f"\nLoading saved heads from {output_dir} ...")
        projector = load_heads(output_dir, device)
        assert projector.pool_cfg is None or projector.pool_cfg.get("mode") == "last", \
            f"checkpoint was trained with pooling {projector.pool_cfg} — " \
            "this script now only supports last-token heads"

    cfg = DATASET_REGISTRY[datasets[-1]]
    print(f"\nLoading test data from {cfg['test']} ...")
    test_limit   = (args.limit // 5) if args.limit else None
    test_records = load_data(cfg["test"], cfg["label_fn"], limit=test_limit,
                             prompt_field=cfg.get("prompt_field", "prompt"))
    print(f"  {len(test_records):,} test records")

    evaluate(model, tok, test_records, projector, cfg["label_fn"], arch=arch)

    print(f"\n  Training time : {t_train:.1f}s  ({t_train/60:.1f} min)")


if __name__ == "__main__":
    main()
