import copy
import json
import os

import torch
import torch.nn as nn

EELLM_THRESHOLDS = (1.0, 0.9, 0.8, 0.6, 0.4, 0.2)

EELLM_TUNE_HPARAMS = dict(
    global_batch_size=16,
    lr=1e-4,
    min_lr=1e-5,
    lr_warmup_fraction=0.01,
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_eps=1e-5,
    clip_grad=1.0,
    max_seq_len=2048,
)


def default_exit_layers(n_layers):
    return [n_layers // 4 - 1, n_layers // 2 - 1]


def _get_backbone(model, arch):
    if arch == "lg4":
        lm = model.language_model
        return lm.model if hasattr(lm, "model") else lm
    return model.model


def _get_lm_head(model, arch):
    if arch == "lg4":
        lm = model.language_model
        return getattr(lm, "lm_head", None) or model.lm_head
    return model.lm_head


def _get_mlp_block(layer, arch):
    if arch == "shieldgemma":
        return (layer.pre_feedforward_layernorm, layer.mlp,
                layer.post_feedforward_layernorm)
    if arch == "lg4":
        return layer.post_attention_layernorm, layer.feed_forward, None
    return layer.post_attention_layernorm, layer.mlp, None


DEFAULT_EXIT_SIZE = {"mlp_blocks": 1, "mlp_width": 1, "head_hidden": 0}


def exit_size_tag(size):
    s = {**DEFAULT_EXIT_SIZE, **(size or {})}
    if s == DEFAULT_EXIT_SIZE:
        return ""
    tag = f"-mlp{s['mlp_blocks']}x{s['mlp_width']}"
    if s["head_hidden"]:
        tag += f"-hd{s['head_hidden']}"
    return tag


class _MLPBlock(nn.Module):

    def __init__(self, pre_norm, mlp, post_norm):
        super().__init__()
        self.pre_norm  = pre_norm
        self.mlp       = mlp
        self.post_norm = post_norm

    def forward(self, h):
        x = self.mlp(self.pre_norm(h))
        if self.post_norm is not None:
            x = self.post_norm(x)
        return h + x


class ExitHead(nn.Module):

    def __init__(self, pre_norm, mlp, post_norm, exit_norm, out_embed, softcap,
                 extra_blocks=None, head_hidden=0):
        super().__init__()
        self.pre_norm  = pre_norm
        self.mlp       = mlp
        self.post_norm = post_norm
        self.exit_norm = exit_norm
        self.out_embed = out_embed
        self.softcap   = softcap
        if extra_blocks:
            self.extra_blocks = nn.ModuleList(extra_blocks)
        else:
            self.extra_blocks = None
        if head_hidden:
            w = out_embed.weight
            d, V = w.shape[1], w.shape[0]
            self.head_up   = nn.Linear(d, head_hidden, bias=False,
                                       device=w.device, dtype=w.dtype)
            self.head_down = nn.Linear(head_hidden, V, bias=False,
                                       device=w.device, dtype=w.dtype)
            with torch.no_grad():
                nn.init.normal_(self.head_up.weight, std=d ** -0.5)
                self.head_down.weight.zero_()
        else:
            self.head_up = self.head_down = None

    def forward(self, h):
        x = self.mlp(self.pre_norm(h))
        if self.post_norm is not None:
            x = self.post_norm(x)
        x = h + x
        if self.extra_blocks is not None:
            for blk in self.extra_blocks:
                x = blk(x)
        z = self.exit_norm(x)
        logits = self.out_embed(z)
        if self.head_up is not None:
            logits = logits + self.head_down(torch.nn.functional.silu(self.head_up(z)))
        if self.softcap:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits


def _widen_mlp(mlp, factor):
    if factor == 1:
        return mlp
    for name in ("gate_proj", "up_proj", "down_proj"):
        if not isinstance(getattr(mlp, name, None), nn.Linear):
            raise ValueError(f"--exit-mlp-width needs a dense gated MLP; "
                             f"{type(mlp).__name__} has no Linear '{name}'")
    g, u, dn = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    I, D = g.out_features, g.in_features
    J = I * factor

    def _lin(in_f, out_f, like):
        return nn.Linear(in_f, out_f, bias=like.bias is not None,
                         device=like.weight.device, dtype=like.weight.dtype)

    with torch.no_grad():
        new = {}
        for name, src in (("gate_proj", g), ("up_proj", u)):
            lin = _lin(D, J, src)
            std = float(src.weight.float().std())
            lin.weight.normal_(0.0, std)
            lin.weight[:I].copy_(src.weight)
            if src.bias is not None:
                lin.bias.zero_()
                lin.bias[:I].copy_(src.bias)
            new[name] = lin
        lin = _lin(J, D, dn)
        lin.weight.zero_()
        lin.weight[:, :I].copy_(dn.weight)
        if dn.bias is not None:
            lin.bias.copy_(dn.bias)
        new["down_proj"] = lin
    for name, lin in new.items():
        setattr(mlp, name, lin)
    if hasattr(mlp, "intermediate_size"):
        mlp.intermediate_size = J
    return mlp


def build_exit_head(model, arch, layer_idx, layers=None, size=None):
    s = {**DEFAULT_EXIT_SIZE, **(size or {})}
    if layers is None:
        layers = _get_backbone(model, arch).layers
    n_layers = len(layers)

    def _copy_block(li):
        pre, mlp, post = _get_mlp_block(layers[li], arch)
        return (copy.deepcopy(pre),
                _widen_mlp(copy.deepcopy(mlp), s["mlp_width"]),
                copy.deepcopy(post) if post is not None else None)

    pre_norm, mlp, post_norm = _copy_block(layer_idx)
    extra = [_MLPBlock(*_copy_block(min(layer_idx + j, n_layers - 1)))
             for j in range(1, s["mlp_blocks"])]
    backbone = _get_backbone(model, arch)
    cfg_t    = getattr(model.config, "text_config", model.config)
    return ExitHead(
        pre_norm =pre_norm,
        mlp      =mlp,
        post_norm=post_norm,
        exit_norm=copy.deepcopy(backbone.norm),
        out_embed=copy.deepcopy(_get_lm_head(model, arch)),
        softcap  =getattr(cfg_t, "final_logit_softcapping", None),
        extra_blocks=extra,
        head_hidden=s["head_hidden"],
    )


class _RMSNorm(nn.Module):
    def __init__(self, d, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class LMExitHead(nn.Module):
    seq_input = True

    def __init__(self, lm, d_in, eps):
        super().__init__()
        self.in_norm = _RMSNorm(d_in, eps)
        d_lm = lm.config.hidden_size
        self.adapter = nn.Linear(d_in, d_lm, bias=False)
        emb = lm.get_input_embeddings().weight
        with torch.no_grad():
            nn.init.normal_(self.adapter.weight,
                            std=float(emb.float().std()) / d_in ** 0.5)
        self.lm = lm

    def hidden(self, h):
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(0)
        x = self.adapter(self.in_norm(h))
        out = self.lm.model(inputs_embeds=x, use_cache=False).last_hidden_state
        return out[0] if squeeze else out

    def logits(self, x):
        return self.lm.lm_head(x)

    def forward(self, h):
        return self.logits(self.hidden(h))


def is_seq_exit(head):
    return getattr(head, "seq_input", False)


def exit_logits_last(head, h):
    if is_seq_exit(head):
        return head.logits(head.hidden(h)[:, -1, :])
    return head(h[:, -1, :])


def build_lm_exit_head(model, arch, layer_idx, lm_path, random_init=False):
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg_t = getattr(model.config, "text_config", model.config)
    V_guard = _get_lm_head(model, arch).weight.shape[0]
    lm_cfg = AutoConfig.from_pretrained(lm_path)
    if lm_cfg.vocab_size != V_guard:
        raise ValueError(f"LM exit {lm_path} has vocab {lm_cfg.vocab_size}, guard "
                         f"has {V_guard}: the LM must share the guard's tokenizer")
    if random_init:
        lm = AutoModelForCausalLM.from_config(lm_cfg, dtype=torch.float32)
    else:
        lm = AutoModelForCausalLM.from_pretrained(lm_path, dtype=torch.float32)
    lm.config.use_cache = False
    return LMExitHead(lm, cfg_t.hidden_size, getattr(cfg_t, "rms_norm_eps", 1e-5))


def exit_head_flops(head):
    lin = lambda mods: sum(2 * m.in_features * m.out_features
                           for m in mods if isinstance(m, nn.Linear))
    if not is_seq_exit(head):
        return lin(head.modules()), 0
    at_verdict = lin([head.lm.lm_head])
    per_tok = lin(head.modules()) - at_verdict
    return at_verdict, per_tok


def heads_dir_for(model_path):
    name = os.path.basename(model_path.rstrip("/"))
    return os.path.join("models", f"eellm-exits-{name}")


def save_exit_heads(heads, meta, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for li, head in heads.items():
        sd = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
              for k, v in head.state_dict().items()}
        torch.save(sd, os.path.join(out_dir, f"exit_{li}.pt"))
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Exit heads saved → {out_dir}  (layers {sorted(heads)})")


def load_exit_heads(model, arch, heads_dir, device, dtype=torch.bfloat16):
    with open(os.path.join(heads_dir, "config.json")) as f:
        meta = json.load(f)
    heads = {}
    size = meta.get("exit_size")
    for li in meta["exit_layers"]:
        if meta.get("exit_type") == "lm":
            head = build_lm_exit_head(model, arch, li, meta["exit_lm"],
                                      random_init=True)
        else:
            head = build_exit_head(model, arch, li, size=size)
        sd   = torch.load(os.path.join(heads_dir, f"exit_{li}.pt"),
                          map_location="cpu", weights_only=True)
        head.load_state_dict(sd)
        heads[li] = head.to(device=device, dtype=dtype).eval()
    return heads, meta
