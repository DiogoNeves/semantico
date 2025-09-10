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


## Run: runs/mixed_wt2_longer_a05
- Model: distilgpt2
- Dataset: wikitext / wikitext-2-raw-v1 (field: text)
- Block size: 256 | Alpha: 0.5 | Temp: 1.0 | TopK: 256
- Train steps: 192 | Epochs: 2.0 | LR: 5e-05 | BS train/eval: 2/2
- Eval metrics: {"eval_loss": 1.9662392139434814, "eval_ce_loss": 3.604022264480591, "eval_perplexity": 36.7457389831543, "eval_token_acc": 0.3600140056022409, "eval_runtime": 83.3026, "eval_samples_per_second": 0.672, "eval_steps_per_second": 0.336, "epoch": 2.0}
- Test metrics: {"eval_loss": 2.0960004329681396, "eval_ce_loss": 3.849557638168335, "eval_perplexity": 46.9722785949707, "eval_token_acc": 0.3282828282828283, "eval_runtime": 85.0767, "eval_samples_per_second": 0.776, "eval_steps_per_second": 0.388, "epoch": 2.0}

## Longer Run (2 epochs, 300/100/100 docs)
- Val perplexity: baseline=36.723, mixed=36.746 (Δ=+0.023)
- Val token_acc: baseline=0.361, mixed=0.360 (Δ=-0.001)
- Test perplexity: baseline=46.925, mixed=46.972 (Δ=+0.047)
- Test token_acc: baseline=0.327, mixed=0.328 (Δ=+0.001)
- Takeaway: Still parity on CE metrics after longer training; pursue semantic proxy metrics and out-of-domain tests next.


## Run: runs/mixed_wt2_semcheck
- Model: distilgpt2
- Dataset: wikitext / wikitext-2-raw-v1 (field: text)
- Block size: 256 | Alpha: 0.5 | Temp: 1.0 | TopK: 256
- Train steps: 15 | Epochs: 1.0 | LR: 5e-05 | BS train/eval: 2/2
- Eval metrics: {"eval_loss": 2.1199021339416504, "eval_ce_loss": 3.8920490741729736, "eval_perplexity": 49.01121139526367, "eval_token_acc": 0.33130493576741044, "eval_runtime": 15.0002, "eval_samples_per_second": 1.933, "eval_steps_per_second": 1.0, "epoch": 1.0}
- Test metrics: {"eval_loss": 2.2489490509033203, "eval_ce_loss": 4.134416103363037, "eval_perplexity": 62.4531135559082, "eval_token_acc": 0.2873303167420814, "eval_runtime": 15.6518, "eval_samples_per_second": 1.661, "eval_steps_per_second": 0.831, "epoch": 1.0}
- Semantic (SBERT): {"sbert_model": "sentence-transformers/all-MiniLM-L6-v2", "sbert_cosine_mean": 0.46476972103118896, "sbert_cosine_all": [0.31499508023262024, 0.46374642848968506, 0.386816143989563, 0.7131074666976929, 0.1712658405303955, 0.5953035950660706, 0.55072021484375, 0.5270653367042542, 0.3771056830883026, 0.5475714206695557]}
