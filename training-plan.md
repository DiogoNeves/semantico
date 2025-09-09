# Training Plan: Cosine-Distance Objective for Causal LM Fine-Tuning

This plan specifies what we’re testing, why, how to implement it (minimal code), how to run and evaluate it, and how to compare to a standard cross-entropy (CE) baseline. It’s written so a junior developer can implement and validate the experiment.

## 1) Motivation & Hypotheses

Goal: Replace or mix the standard next-token cross-entropy loss with a cosine-distance objective defined in the model’s own token embedding space. We test whether this improves semantic alignment without sacrificing standard LM metrics.

Hypotheses:
- H1: A mixed loss `L = α·L_cos + (1-α)·L_ce` can match or slightly improve validation perplexity versus CE-only on out-of-domain data, for reasonable `α ∈ {0.25, 0.5, 0.75}`.
- H2: Embedding-space alignment improves. Proxy metric (embedding-space): average cosine between predicted expected embedding and the gold token embedding increases.
- H3: Pure cosine loss (α=1.0) may be unstable early in training; mixing with CE stabilizes optimization.

Why this might help: CE encourages probability mass on the exact gold token, while cosine in embedding space rewards placing mass on tokens with similar embeddings (near-synonyms, morphology). Mixing may yield better semantic robustness.

## 2) Theory: Objective Definitions

Setup: For a position t, model outputs logits `z_t ∈ R^V`, probabilities `p_t = softmax(z_t / T)`, where V is vocabulary size and T is a temperature. Let `E ∈ R^{V×H}` be the model’s tied input embedding matrix (H is hidden size). The gold token is `y_t`.

- Expected predicted embedding: `μ_t = Σ_v p_t[v] · E[v]` (matrix form: `μ = p @ E`).
- Gold embedding: `e_t = E[y_t]`.
- Cosine loss per token: `L_cos(t) = 1 - cosine( μ_t / ||μ_t||, e_t / ||e_t|| )`.
- Sequence loss: mean of valid positions (ignore index = -100).
- Mixed loss: `L = α·L_cos + (1-α)·L_ce`, where `L_ce` is standard next-token CE (with logits shifted by one position). Hyperparameters: `α ∈ [0,1]`, `T > 0`.

Notes:
- We use the LM’s own embedding matrix E for both μ and e. This keeps everything differentiable and consistent.
- Optional efficiency: approximate `μ_t` using top-k of `p_t` (e.g., k=256) to reduce compute/memory.

## 3) Data & Leakage Considerations

Question: Has `distilgpt2` “seen” `wikitext-2-raw-v1`? DistilGPT2 is distilled from GPT-2 (trained on WebText, not Wikitext-2 specifically). However, content overlap with Wikipedia is plausible. For a clean evaluation story, prefer data unlikely to be in GPT-2’s pretraining distribution.

Recommendations (pick one):
- Option A (preferred by you): `HuggingFaceFW/finepdfs` (PDF text domain). Use a subset for speed. If this dataset isn’t available locally, fall back to options below.
- Option B: Domain-specific corpora (e.g., scientific papers). Example on HF: arXiv/papers datasets (choose a plain-text field). Another option is `ccdv/arxiv-summarization` (use the `article` field as raw text).
- Option C: `bookcorpusopen` (books; still likely different distribution from WebText).

Splitting & leakage avoidance:
- Split by document (not by random lines) into train/val/test (80/10/10 or use provided splits). Do not allow the same document to appear across splits.
- Tokenize each split independently after splitting.

Minimal subset for quick iteration:
- Train: 5k documents (or ~50–100M tokens if available)
- Val: 500 documents
- Test: 500 documents

Text cleaning (lightweight):
- Drop empty or very short docs (< 128 characters).
- Normalize whitespace, strip repeated newlines if PDFs are noisy.

## 4) Experimental Design (What We’ll Run)

Models:
- Primary: `distilgpt2` (fast). Risk of partial overlap with pretraining corpus is acceptable for ablations since all variants share the same base model and data.
- Optional: From-scratch tiny GPT-2 config for a leakage-free comparison (≈ 10–20M params). Use this only after the fine-tune prototype works.

Loss variants:
- Baseline: CE-only (α=0.0).
- Mixed: α ∈ {0.25, 0.5, 0.75}.
- Cos-only: α=1.0 (exploratory; watch for instability).

Temperatures: T ∈ {1.0, 0.7, 1.5}.

Hyperparameters (starting point):
- Max length (`block_size`): 512
- Batch size (per device): 8–16 (adjust for memory)
- LR: 5e-5 (fine-tuning), weight_decay: 0.01, warmup_ratio: 0.03
- Epochs: 1–3 for quick comparison runs
- Grad clip: 1.0
- FP16: enabled if GPU supports it

Seeds: Run at least 3 seeds for the main comparison if resources allow.

## 5) Implementation Plan (Minimal Code)

Files (keep it simple):
- `train.py`: CLI, data loading, tokenizer, Trainer subclass with custom `compute_loss`, eval metrics, run loop.
- `losses.py`: cosine loss function utilities.

Environment setup:
- Python 3.10+
- `pip install transformers datasets accelerate torch evaluate` (plus `bitsandbytes` if you want 8-bit loading on GPU). Optionally `wandb` for logging.

Tokenizer & padding:
- Use the base model tokenizer: `AutoTokenizer.from_pretrained(model_name)`.
- For GPT-2 family set pad token: `tokenizer.pad_token = tokenizer.eos_token` to enable padding in batches.

Data loading outline (Hugging Face Datasets):
1) Load dataset
```
from datasets import load_dataset
ds = load_dataset(args.dataset_name, args.dataset_config_name or None)
# If the dataset has no predefined splits, manually split the "train" into train/val/test by document.
```
2) Select text field
```
text_col = args.text_field  # e.g., "text", "content", or "article"
```
3) Tokenize and group into fixed-length blocks
```
def tokenize_fn(batch):
    return tokenizer(batch[text_col], add_special_tokens=False)

tokenized = ds.map(tokenize_fn, batched=True, remove_columns=[c for c in ds["train"].column_names if c != text_col])

def group_texts(examples):
    # Concatenate and chunk by block_size
    concatenated = []
    for ids in examples["input_ids"]:
        concatenated.extend(ids)
    total_length = (len(concatenated) // args.block_size) * args.block_size
    input_ids = [
        concatenated[i : i + args.block_size]
        for i in range(0, total_length, args.block_size)
    ]
    return {"input_ids": input_ids, "labels": input_ids.copy()}

lm_ds = tokenized.map(group_texts, batched=True)
```

Cosine loss utility (`losses.py`):
```
import torch
import torch.nn.functional as F

def cosine_embedding_loss_from_logits(logits, labels, embed, temperature=1.0, topk=None):
    # logits: [B, T, V], labels: [B, T], embed: [V, H]
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(-100)
    if not mask.any():
        return logits.new_tensor(0.0)

    probs = F.softmax(shift_logits / temperature, dim=-1)  # [B, T-1, V]
    E = embed.to(probs.dtype)  # [V, H]

    if topk is not None and topk > 0 and topk < probs.size(-1):
        vals, idx = torch.topk(probs, k=topk, dim=-1)  # [B, T-1, K]
        gathered = F.embedding(idx, E)  # [B, T-1, K, H]
        expected = torch.einsum("btk,btkh->bth", vals, gathered)  # [B, T-1, H]
    else:
        expected = probs @ E  # [B, T-1, H]

    target = F.embedding(shift_labels.clamp_min(0), E)  # [B, T-1, H]
    expected = F.normalize(expected, dim=-1)
    target = F.normalize(target, dim=-1)
    cos = (expected * target).sum(dim=-1)  # [B, T-1]
    loss = 1.0 - cos
    loss = (loss * mask).sum() / mask.sum()
    return loss
```

Trainer subclass (`train.py` core):
```
import torch
import torch.nn.functional as F
from transformers import Trainer
from losses import cosine_embedding_loss_from_logits

class CosineTrainer(Trainer):
    def __init__(self, *args, alpha=0.5, temperature=1.0, cos_topk=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.cos_topk = cos_topk

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, V]
        vocab_size = logits.size(-1)

        # CE part (standard next-token CE)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Cosine part (embedding space)
        embed = model.get_input_embeddings().weight  # [V, H]
        cos_loss = cosine_embedding_loss_from_logits(
            logits, labels, embed, temperature=self.temperature, topk=self.cos_topk
        )

        # Mixed loss
        loss = self.alpha * cos_loss + (1.0 - self.alpha) * ce_loss
        return (loss, outputs) if return_outputs else loss
```

Metrics (perplexity, accuracy, proxy embedding metric):
```
import evaluate
import torch
import torch.nn.functional as F

def build_compute_metrics(tokenizer, block_size):
    def compute_metrics_fn(eval_pred):
        # eval_pred.predictions: [N, T, V] or tuple; labels: [N, T]
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]
        logits = torch.tensor(preds)
        labels = torch.tensor(labels)
        V = logits.size(-1)

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce = F.cross_entropy(
            shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100, reduction="mean"
        ).item()
        ppl = float(torch.exp(torch.tensor(ce)))

        # Token accuracy (top-1)
        preds_top1 = shift_logits.argmax(dim=-1)
        mask = shift_labels.ne(-100)
        correct = (preds_top1.eq(shift_labels) & mask).sum().item()
        total = mask.sum().item() or 1
        acc = correct / total

        return {"ce_loss": ce, "perplexity": ppl, "token_acc": acc}

    return compute_metrics_fn
```

CLI flags (suggested):
- `--model_name` (default: `distilgpt2`)
- `--dataset_name`, `--dataset_config_name`, `--text_field`
- `--subset_train`, `--subset_val`, `--subset_test` (optional limits)
- `--block_size` (default: 512)
- `--alpha`, `--temperature`, `--cos_topk`
- `--from_scratch` (bool) to build a tiny GPT-2 config if desired

Optional: From-scratch tiny model
```
from transformers import GPT2Config, GPT2LMHeadModel

if args.from_scratch:
    cfg = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.block_size,
        n_embd=512,
        n_layer=4,
        n_head=8,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(cfg)
else:
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
```

## 6) Runbook (Baseline and Variants)

Baseline CE-only:
```
python train.py \
  --model_name distilgpt2 \
  --dataset_name HuggingFaceFW/finepdfs \
  --text_field text \
  --block_size 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --alpha 0.0 \
  --temperature 1.0
```

Mixed (α=0.5):
```
python train.py \
  --model_name distilgpt2 \
  --dataset_name HuggingFaceFW/finepdfs \
  --text_field text \
  --block_size 512 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --num_train_epochs 1 \
  --learning_rate 5e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.03 \
  --alpha 0.5 \
  --temperature 1.0
```

Try α ∈ {0.25, 0.75} and T ∈ {0.7, 1.5}. If memory is tight, add `--cos_topk 256`.

If `HuggingFaceFW/finepdfs` is unavailable, substitute:
- `--dataset_name bookcorpusopen` and `--text_field text`, or
- `--dataset_name ccdv/arxiv-summarization` and `--text_field article`.

## 7) Validation & Checks

Quick functional checks (first run with ~1k documents):
- Loss decreasing: Ensure total loss and CE both decrease on training.
- Eval metrics: Perplexity and token accuracy compute without error; compare baseline vs mixed.
- Sanity generations: Sample 5–10 prompts from the validation set; verify outputs are coherent.

Logged metrics to track each step:
- `train_loss`, `eval_loss` (mixed or cosine-only loss depending on α)
- `eval/ce_loss`, `eval/perplexity`, `eval/token_acc`
- Optional: cosine proxy metric (mean cosine between μ and e on eval set) if added to `compute_metrics`.

Reproducibility:
- Fix seed (`--seed 42`); set `torch.backends.cudnn.deterministic = True` (with potential perf tradeoff).
- Keep data splits and preprocess deterministic; log exact dataset name/config and text field.

## 8) Analysis & Reporting

Report per variant (baseline, α=0.25/0.5/0.75, α=1.0):
- Validation perplexity, token accuracy.
- Training stability observations (divergence, gradient spikes, speed difference due to cosine).
- Qualitative examples (side-by-side generations). If the claim is about semantics, show paraphrase-like improvements.

Decision criteria:
- If mixed loss equals or improves perplexity and/or token accuracy while improving semantic proxy metrics, consider it a win.
- If perplexity worsens modestly but qualitative semantics improve, document the trade-off; consider adding a semantic sentence-level metric (optional follow-up).

## 9) Risks & Mitigations

- Instability with α=1.0: Start with α=0.25–0.5; optionally ramp α from 0.0 to target over first 10% steps.
- Memory/compute overhead: `μ = p @ E` costs O(V·H); mitigate with `--cos_topk`.
- Data leakage concerns: Use out-of-domain data (e.g., PDFs, scientific text) and focus on same-data comparisons across variants.
- Eval mismatch: Cosine may optimize semantics while CE-based perplexity doesn’t capture it. Mitigate with additional proxy metrics and qualitative review.

## 10) Extensions (Future Work)

- External semantic targets: Use Sentence-BERT to compute target embeddings for entire tokens or spans; requires vocab alignment or nearest-neighbor mapping.
- Predict embeddings directly: Remove LM head CE entirely; train to predict target embeddings and decode via nearest-neighbor (ANN). Slower decoding; larger change.
- Curriculum: Start with CE-only, gradually increase α.
- Alternative geometries: Contrastive objectives bringing predicted μ closer to e and pushing away random negatives.

## 11) Deliverables

- Code: `train.py`, `losses.py` with flags above.
- Artifacts: Best checkpoint, logs, and a `metrics.json` summarizing runs.
- Short report: 1–2 pages comparing baseline vs cosine variants with plots (loss curves, perplexity over steps).

---

FAQs (from your questions):

- Has `distilgpt2` seen `wikitext-2-raw-v1`? Not directly, but overlap with Wikipedia is plausible. Using out-of-domain data (like PDFs) is better for claims; for method iteration, Wikitext-2 is fine if you compare variants fairly.
- Can we still use tokens? Yes. Keep the same tokenizer and vocabulary. The cosine objective uses the model’s token embedding space; no new discretization is needed for this experiment.
- How do we split text? Split by document first; then tokenize and concatenate within each split; then chunk into fixed-length windows (`block_size`) and set `labels = input_ids` (shift is handled in the loss).
- Do we need BERT? No. Start with the model’s own embeddings. External embeddings (BERT/SBERT) are an optional future extension.

