import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

from losses import cosine_embedding_loss_from_logits


@dataclass
class Args:
    model_name: str
    dataset_name: str
    dataset_config_name: Optional[str]
    text_field: str
    subset_train: Optional[int]
    subset_val: Optional[int]
    subset_test: Optional[int]
    block_size: int
    alpha: float
    temperature: float
    cos_topk: Optional[int]
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    num_train_epochs: float
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    gradient_accumulation_steps: int
    output_dir: str
    from_scratch: bool
    seed: int
    fp16: bool
    eval_strategy: str
    logging_steps: int
    save_total_limit: int
    min_chars: int
    sample_generations: int
    generation_max_new_tokens: int
    semantic_eval_samples: int
    semantic_metric: str
    sbert_model: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Causal LM fine-tuning with cosine loss option")
    p.add_argument("--model_name", default="distilgpt2")
    p.add_argument("--dataset_name", required=True)
    p.add_argument("--dataset_config_name", default=None)
    p.add_argument("--text_field", required=True)
    p.add_argument("--subset_train", type=int, default=None)
    p.add_argument("--subset_val", type=int, default=None)
    p.add_argument("--subset_test", type=int, default=None)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--cos_topk", type=int, default=None)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=8)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--output_dir", default="runs/exp")
    p.add_argument("--from_scratch", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--eval_strategy", default="epoch", choices=["no", "steps", "epoch"])
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--min_chars", type=int, default=128)
    p.add_argument("--sample_generations", type=int, default=0, help="number of val samples to generate")
    p.add_argument("--generation_max_new_tokens", type=int, default=64)
    p.add_argument("--semantic_eval_samples", type=int, default=0, help="number of val samples for semantic eval")
    p.add_argument("--semantic_metric", default="sbert", choices=["sbert", "none"], help="semantic metric to compute on generations")
    p.add_argument("--sbert_model", default="sentence-transformers/all-MiniLM-L6-v2")
    a = p.parse_args()
    return Args(**vars(a))


class CosineTrainer(Trainer):
    def __init__(self, *args, alpha: float = 0.5, temperature: float = 1.0, cos_topk: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self.cos_topk = cos_topk

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch: Optional[int] = None):
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits  # [B, T, V]
        vocab_size = logits.size(-1)

        # CE loss
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        # Cosine loss
        embed = model.get_input_embeddings().weight  # [V, H]
        cos_loss = cosine_embedding_loss_from_logits(
            logits, labels, embed, temperature=self.temperature, topk=self.cos_topk
        )

        loss = self.alpha * cos_loss + (1.0 - self.alpha) * ce_loss
        return (loss, outputs) if return_outputs else loss


def build_compute_metrics():
    def compute_metrics_fn(eval_pred):
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

        preds_top1 = shift_logits.argmax(dim=-1)
        mask = shift_labels.ne(-100)
        correct = (preds_top1.eq(shift_labels) & mask).sum().item()
        total = int(mask.sum().item()) or 1
        acc = correct / total

        return {"ce_loss": ce, "perplexity": ppl, "token_acc": acc}

    return compute_metrics_fn


def ensure_splits(ds: DatasetDict, rng_seed: int) -> DatasetDict:
    if set(ds.keys()) >= {"train", "validation", "test"}:
        return ds
    if "train" in ds and "validation" in ds and "test" not in ds:
        # create test from validation
        val = ds["validation"].train_test_split(test_size=0.5, seed=rng_seed)
        return DatasetDict(train=ds["train"], validation=val["train"], test=val["test"])
    if "train" in ds and "validation" not in ds and "test" not in ds:
        split = ds["train"].train_test_split(test_size=0.2, seed=rng_seed)
        val_test = split["test"].train_test_split(test_size=0.5, seed=rng_seed)
        return DatasetDict(train=split["train"], validation=val_test["train"], test=val_test["test"])
    # Fallback: if only a single key exists, treat it as train
    only_key = next(iter(ds.keys()))
    split = ds[only_key].train_test_split(test_size=0.2, seed=rng_seed)
    val_test = split["test"].train_test_split(test_size=0.5, seed=rng_seed)
    return DatasetDict(train=split["train"], validation=val_test["train"], test=val_test["test"])


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        # GPT-2 family needs pad token
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    ds = load_dataset(args.dataset_name, args.dataset_config_name)  # may return DatasetDict or dict-like
    ds = ensure_splits(ds, args.seed)

    # Drop short docs and clean whitespace (light cleaning)
    def clean_filter(example):
        txt = example.get(args.text_field, "")
        if not isinstance(txt, str):
            return False
        s = " ".join(txt.split())
        example[args.text_field] = s
        return len(s) >= args.min_chars

    ds = ds.map(lambda x: {args.text_field: " ".join(str(x[args.text_field]).split())})
    ds = ds.filter(clean_filter)

    # Apply subsets if requested (by document, before tokenization)
    def head_if(ds_split, n):
        if n is None:
            return ds_split
        n = min(n, ds_split.num_rows)
        return ds_split.select(range(n))

    ds = DatasetDict(
        train=head_if(ds["train"], args.subset_train),
        validation=head_if(ds["validation"], args.subset_val),
        test=head_if(ds["test"], args.subset_test),
    )

    # Tokenize
    def tokenize_fn(batch):
        return tokenizer(batch[args.text_field], add_special_tokens=False)

    remove_cols_train = [c for c in ds["train"].column_names if c != args.text_field]
    remove_cols_val = [c for c in ds["validation"].column_names if c != args.text_field]
    remove_cols_test = [c for c in ds["test"].column_names if c != args.text_field]

    tokenized = DatasetDict(
        train=ds["train"].map(tokenize_fn, batched=True, remove_columns=remove_cols_train),
        validation=ds["validation"].map(tokenize_fn, batched=True, remove_columns=remove_cols_val),
        test=ds["test"].map(tokenize_fn, batched=True, remove_columns=remove_cols_test),
    )

    # Group into fixed-length sequences
    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)
        total_length = (len(concatenated) // args.block_size) * args.block_size
        input_ids = [
            concatenated[i : i + args.block_size]
            for i in range(0, total_length, args.block_size)
        ]
        if len(input_ids) == 0:
            return {"input_ids": [], "labels": []}
        return {"input_ids": input_ids, "labels": input_ids.copy()}

    lm_ds = DatasetDict(
        train=tokenized["train"].map(
            group_texts, batched=True, remove_columns=tokenized["train"].column_names
        ),
        validation=tokenized["validation"].map(
            group_texts, batched=True, remove_columns=tokenized["validation"].column_names
        ),
        test=tokenized["test"].map(
            group_texts, batched=True, remove_columns=tokenized["test"].column_names
        ),
    )

    # Model
    if args.from_scratch:
        cfg = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=args.block_size,
            n_embd=512,
            n_layer=4,
            n_head=8,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        model = GPT2LMHeadModel(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy=args.eval_strategy,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        save_strategy="no",  # disable checkpoints for quicker iterations by default
        report_to=[],  # disable W&B by default; can be enabled by env
        fp16=args.fp16 and torch.cuda.is_available(),
    )

    trainer = CosineTrainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=lm_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(),
        alpha=args.alpha,
        temperature=args.temperature,
        cos_topk=args.cos_topk,
    )

    # Train and evaluate
    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    # Test set eval for completeness (optional)
    try:
        test_metrics = trainer.evaluate(eval_dataset=lm_ds["test"])  # may be similar to val
    except Exception:
        test_metrics = {}

    # Optional: sample generations from validation set
    gens = []
    if args.sample_generations > 0 and len(lm_ds["validation"]) > 0:
        model.eval()
        rng = random.Random(args.seed)
        indices = [rng.randrange(0, len(lm_ds["validation"])) for _ in range(args.sample_generations)]
        for idx in indices:
            item = lm_ds["validation"][idx]
            # use first half of the block as prompt
            input_ids = item["input_ids"][: max(1, args.block_size // 2)]
            input_ids = torch.tensor([input_ids])
            input_ids = input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.generation_max_new_tokens,
                    do_sample=True,
                    temperature=0.9,
                    top_k=50,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
            gens.append({"index": idx, "prompt": prompt, "generation": text})

        with open(os.path.join(args.output_dir, "generations.json"), "w") as f:
            json.dump(gens, f, indent=2)

    # Optional: semantic evaluation using SBERT on generated continuations vs gold
    semantic = {}
    if args.semantic_metric == "sbert" and args.semantic_eval_samples > 0 and len(lm_ds["validation"]) > 0:
        try:
            from sentence_transformers import SentenceTransformer, util

            model.eval()
            sbert = SentenceTransformer(args.sbert_model)
            rng = random.Random(args.seed)
            indices = [rng.randrange(0, len(lm_ds["validation"])) for _ in range(args.semantic_eval_samples)]
            preds, refs = [], []
            for idx in indices:
                item = lm_ds["validation"][idx]
                input_ids_full = item["input_ids"]
                split = max(1, len(input_ids_full) // 2)
                prompt_ids = input_ids_full[:split]
                gold_ids = input_ids_full[split: split + args.generation_max_new_tokens]
                prompt = torch.tensor([prompt_ids], device=model.device)
                with torch.no_grad():
                    out = model.generate(
                        input_ids=prompt,
                        max_new_tokens=len(gold_ids) if len(gold_ids) > 0 else args.generation_max_new_tokens,
                        do_sample=True,
                        temperature=0.9,
                        top_k=50,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
                gen_text = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
                ref_text = tokenizer.decode(gold_ids, skip_special_tokens=True)
                preds.append(gen_text)
                refs.append(ref_text)

            emb_pred = sbert.encode(preds, convert_to_tensor=True, show_progress_bar=False)
            emb_ref = sbert.encode(refs, convert_to_tensor=True, show_progress_bar=False)
            from torch.nn import functional as Fnn
            cos = Fnn.cosine_similarity(emb_pred, emb_ref).cpu().numpy().tolist()
            semantic = {"sbert_model": args.sbert_model, "sbert_cosine_mean": float(sum(cos) / max(1, len(cos))), "sbert_cosine_all": cos}
        except Exception as e:
            semantic = {"error": str(e)}

    # Persist metrics (now including optional semantic)
    metrics = {
        "args": vars(args),
        "eval": eval_metrics,
        "test": test_metrics,
        "train": {k: float(v) if isinstance(v, (int, float)) else v for k, v in train_result.metrics.items()},
        "semantic": semantic,
    }
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Append to results.md for quick tracking
    results_path = os.path.join(os.getcwd(), "results.md")
    with open(results_path, "a") as rf:
        rf.write("\n\n## Run: {}\n".format(args.output_dir))
        rf.write("- Model: {}\n".format(args.model_name))
        rf.write("- Dataset: {} / {} (field: {})\n".format(args.dataset_name, args.dataset_config_name, args.text_field))
        rf.write("- Block size: {} | Alpha: {} | Temp: {} | TopK: {}\n".format(args.block_size, args.alpha, args.temperature, args.cos_topk))
        rf.write("- Train steps: {} | Epochs: {} | LR: {} | BS train/eval: {}/{}\n".format(
            train_result.global_step if hasattr(train_result, "global_step") else "?",
            args.num_train_epochs,
            args.learning_rate,
            args.per_device_train_batch_size,
            args.per_device_eval_batch_size,
        ))
        rf.write("- Eval metrics: {}\n".format(json.dumps(eval_metrics)))
        if test_metrics:
            rf.write("- Test metrics: {}\n".format(json.dumps(test_metrics)))
        if gens:
            rf.write("- Sample generations saved to {}/generations.json\n".format(args.output_dir))
        if semantic:
            rf.write("- Semantic (SBERT): {}\n".format(json.dumps(semantic)))


if __name__ == "__main__":
    # Allow both python -m and python train.py usage
    main()
