# SafeCache

SafeCache, EE-LLM and GPTCache on Llama Guard 3/4, ShieldGemma and Aegis Guard.
Tested on one 48 GB GPU (RTX A6000), CUDA 13.0 driver.

## Setup

```bash
conda env create -f environment.yml
conda activate safecache
```

Base guards (needs a Hugging Face login with the licenses of
`meta-llama/Llama-Guard-3-8B`, `meta-llama/Llama-Guard-4-12B`, `google/shieldgemma-9b`,
`meta-llama/LlamaGuard-7b` and `nvidia/Aegis-AI-Content-Safety-LlamaGuard-Permissive-1.0`
accepted):

```bash
hf auth login
bash scripts/download_models.sh
```

Datasets and trained components (LoRA adapters, projection heads, EE-LLM exits):
download the files below, place them in the repository root and extract them there.

| File | Content | Size |
|------|---------|------|
| [safecache_data.zip](GDRIVE_LINK_DATA) | datasets → `data/` | 51 MB |
| [safecache_lg3.zip](GDRIVE_LINK_LG3) | Llama Guard 3 components → `models/` | 2.9 GB |
| [safecache_lg4.zip](GDRIVE_LINK_LG4) | Llama Guard 4 components → `models/` | 4.8 GB |
| [safecache_sg.zip](GDRIVE_LINK_SG) | ShieldGemma components → `models/` | 4.2 GB |
| [safecache_ag.zip](GDRIVE_LINK_AG) | Aegis Guard components → `models/` | 1.7 GB |

```bash
unzip safecache_data.zip
unzip safecache_lg3.zip
unzip safecache_lg4.zip
unzip safecache_sg.zip
unzip safecache_ag.zip
```

Only the zips of the guards you run are needed.

## Run

```bash
bash scripts/run_safecache.sh
bash scripts/run_eellm.sh
bash scripts/run_gptcache.sh
```

For a subset of guards (`lg3`, `lg4`, `sg`, `ag`):

```bash
GUARDS="lg3" bash scripts/run_safecache.sh
```

Results:

```
results/result-supcon-frac30/results_sweep_{guard}-supcon-frac30_{dataset}.csv
results/result-eellm/results_sweep_{guard}_eellm_fine_{dataset}.csv
results/result-gptcache/results_gptcache_{guard}_{dataset}.csv
```

In each CSV, the `lg_only` row is the guard alone (Base). The reported operating point
is the row with the lowest `ms_avg` whose `f1` is at most 0.02 below `lg_only`. For
EE-LLM and GPTCache, only thresholds 0.70, 0.73, …, 0.97 are considered.

