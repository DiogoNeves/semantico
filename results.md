# Experiment Results

This file collects results and observations from training runs.

- Initialized on first run.

Observations (initial quick run on Wikitext-2 small subset):
- Mixed loss (alpha=0.5, topk=256) runs stably on CPU and adds compute overhead vs CE-only.
- CE-based metrics (perplexity, token_acc) are nearly identical between baseline and mixed on this small setting.
- Mixed loss eval_loss is lower by construction (includes cosine term); compare using eval_ce_loss/perplexity.
- Using Wikitext-2 for iteration; can switch to PDFs/arXiv per plan for out-of-domain claims.

## Conclusion
- Pursue: Early results show no regression in CE metrics when adding cosine loss (α=0.5), suggesting the approach is viable. Compute overhead is noticeable but manageable with `--cos_topk`.
- Next: Run α in {0.25, 0.5, 0.75} and T in {0.7, 1.5}; compute an embedding-space cosine proxy metric; try out-of-domain data (PDFs/arXiv). If CE remains flat and the proxy improves, adopt mixed loss (with top-k) for robustness; avoid α=1.0 initially or ramp α.


## Run: runs/baseline_wt2_small
- Model: distilgpt2
- Dataset: wikitext / wikitext-2-raw-v1 (field: text)
- Block size: 256 | Alpha: 0.0 | Temp: 1.0 | TopK: None
- Train steps: 28 | Epochs: 1.0 | LR: 5e-05 | BS train/eval: 2/2
- Eval metrics: {"eval_loss": 3.845308303833008, "eval_ce_loss": 3.845276355743408, "eval_perplexity": 46.7716064453125, "eval_token_acc": 0.33644354293441514, "eval_runtime": 9.7006, "eval_samples_per_second": 2.989, "eval_steps_per_second": 1.546, "epoch": 1.0}
- Test metrics: {"eval_loss": 4.09746789932251, "eval_ce_loss": 4.097436904907227, "eval_perplexity": 60.185829162597656, "eval_token_acc": 0.29321266968325793, "eval_runtime": 9.3548, "eval_samples_per_second": 2.779, "eval_steps_per_second": 1.39, "epoch": 1.0}


## Run: runs/mixed_wt2_small_a05
- Model: distilgpt2
- Dataset: wikitext / wikitext-2-raw-v1 (field: text)
- Block size: 256 | Alpha: 0.5 | Temp: 1.0 | TopK: 256
- Train steps: 28 | Epochs: 1.0 | LR: 5e-05 | BS train/eval: 2/2
- Eval metrics: {"eval_loss": 2.09562087059021, "eval_ce_loss": 3.8456614017486572, "eval_perplexity": 46.78961944580078, "eval_token_acc": 0.3373901284651792, "eval_runtime": 13.052, "eval_samples_per_second": 2.222, "eval_steps_per_second": 1.149, "epoch": 1.0}
- Test metrics: {"eval_loss": 2.229314088821411, "eval_ce_loss": 4.097716808319092, "eval_perplexity": 60.202674865722656, "eval_token_acc": 0.2942684766214178, "eval_runtime": 13.647, "eval_samples_per_second": 1.905, "eval_steps_per_second": 0.953, "epoch": 1.0}

## Baseline vs Mixed (quick check)
- Val perplexity: baseline=46.772, mixed=46.790 (Δ=+0.018)
- Val token_acc: baseline=0.336, mixed=0.337 (Δ=+0.001)
- Test perplexity: baseline=60.186, mixed=60.203 (Δ=+0.017)
- Test token_acc: baseline=0.293, mixed=0.294 (Δ=+0.001)
- Takeaway: On this tiny run, CE metrics are essentially equal; need more data/seeds to conclude.


## Run: runs/baseline_wt2_longer
- Model: distilgpt2
- Dataset: wikitext / wikitext-2-raw-v1 (field: text)
- Block size: 256 | Alpha: 0.0 | Temp: 1.0 | TopK: None
- Train steps: 192 | Epochs: 2.0 | LR: 5e-05 | BS train/eval: 2/2
- Eval metrics: {"eval_loss": 3.6034348011016846, "eval_ce_loss": 3.603400707244873, "eval_perplexity": 36.72290802001953, "eval_token_acc": 0.3608543417366947, "eval_runtime": 92.605, "eval_samples_per_second": 0.605, "eval_steps_per_second": 0.302, "epoch": 2.0}
- Test metrics: {"eval_loss": 3.8485865592956543, "eval_ce_loss": 3.848552703857422, "eval_perplexity": 46.92509841918945, "eval_token_acc": 0.32727272727272727, "eval_runtime": 130.4711, "eval_samples_per_second": 0.506, "eval_steps_per_second": 0.253, "epoch": 2.0}
