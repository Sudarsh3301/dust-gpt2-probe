# Dust: distillation

Paper: "Dust: Pretraining Transformers Without Backpropagation", Dahal, Mandal, Gülbahar, Vegesna (qlabs), October 2026. https://qlabs.sh/research/dust
Code: https://github.com/qlabs-eng/dust @ 2fdb01b (2026-10-04), MIT, about 1.3k lines.

Grades: [source:code] means read in the repo. [source:page] means read on the paper page; the page is blocked from the workspace shell, so it was read through WebFetch's summarizer and the exact numbers still need a direct read. [unverified] means a snippet or an estimate.

## Mechanism in one sentence
Instead of perturbing weights (one forward pass per population member), perturb each linear layer's *output* with independent Gaussian noise at every token. Each token is then a "virtual" population member, so one forward pass scores B·T perturbations. The reward-weighted noise is the error estimate at that layer's output, and the weight gradient is its outer product with the cached input, exactly as in backprop. [source:code dust.py:175-199, 325-331]

## How it works in the code [source:code]
- **Clean pass.** One clean forward caches every Linear/Embedding input and output and the per-token losses.
- **Direct draws.** For the attention and MLP output projections (perturbed jointly per block), mlp.c_fc, the token embedding, and the attention output before c_proj:
  - add σ·a, with σ = 0.2 (0.4 for the attention output in the deeper half);
  - rerun the forward pass;
  - reward = clean loss − perturbed loss, **at the same token only**;
  - centre the reward across the nrep draws in a chunk;
  - error estimate = −mean(reward·a)/σ.
- **Attention internals** (q, k, v, value-embedding gate, value embeddings): perturb, recompute only that block's attention, and score Δo against the *estimated* error at the attention output. For k, v, gate and value embeddings, the scores of later tokens are summed with γ^lag (γ = 0.98, or 0.99 at 10M tokens / 16k draws); q keeps only its own token.
- **Head:** jittered on the cached logits, one 256-column vocabulary slab at a time, with 4·K draws.
- **Residual-mix scalars** (2L of them): trained with two-sided weight-space ES, 8 draws.
- **Optimizer:** SGD with momentum and a constant LR. The embedding LR is 1000 (same as the backprop baseline).
- **Population K** counts direct-loss draws only. At K = 256: writers 110 + c_fc 110 + attention outputs 26 + embedding 10.
- **Paper wording:** r_t = Σ_{s≥t} γ^{s−t} c_s, with γ = 0 for the direct sites and γ close to 1 for the attention internals [source:page]. This is consistent with the code.

## Claims table
| # | Claim | Where | Setting | Baseline | Compute-matched? |
|---|---|---|---|---|---|
| 1 | Approaches, and sometimes exceeds, backprop | Fig 2 / Tab 1 [source:page] | 8L×512 (~25M non-embedding), FineWeb, BPE-4096, 16k-token batches, 1 epoch, 3 seeds | backprop with SGD+momentum, constant LR, same grid | **No.** At equal tokens Dust uses roughly K forward passes per step, versus about 3 for backprop |
| 1a | At 1M tokens: Dust@16k ≈ 5.03 vs backprop ≈ 5.07 | Tab 1 [source:page, via summarizer — verify] | 61 steps | same | No (~16k forward passes per step vs ~3) |
| 1b | At 10M tokens: Dust@16k ≈ 5.036 vs backprop 5.015, so not exceeded | Tab 1 [source:page] | 610 steps | same | No |
| 1c | At 20M tokens: fitted limit 4.431 (95% CI 3.89–4.58) vs backprop 4.633 | Fig 2 [source:page] | **an extrapolated limit, not a measured number** | same | No |
| 1d | With Adam at 1M tokens: Dust limit 5.248 (CI 5.17–5.30) vs backprop 5.361 | Fig 3 [source:page] | fitted limit | Adam backprop | No |
| 2 | 1,000–10,000× more population-efficient than EGGROLL | Fig 2 [source:page] | EGGROLL at 16k doesn't reach Dust at 64 | EGGROLL | Population-matched only, not FLOP-matched |
| 3 | Bigger is better at fixed population | Fig 4 [source:page] | 2M/7M/38M/243M parameters, 10M tokens, K 64–16k | across model sizes | n/a |
| 4 | Cosine with the backprop gradient follows c_max/√(1+c/K) (RMSE < 0.06) from 10M to 1B tokens | Fig 5 [source:page] | single-batch cosine | EGGROLL cosine < 0.05 at 128k | n/a |

## Author thread claims (pasted by user, 2026-10-06)
- "Dust beats backprop at 100k and 1M tokens." This conflicts with the page summary, which gives 100k as Dust ≈ 5.24 vs backprop ≈ 5.20. [conflict — read Table 1 directly before citing]
- "Dust's gradients don't converge to backprop's exactly. They end up useful but different, and empirically that's sometimes better." The authors admit the K→∞ limit is not the backprop gradient, but they do not say what it is. Direction 1 tests our candidate answer: a Gaussian-smoothed loss with same-token truncation.
- "Overparameterization helps by giving search a bigger space with better geometry." A cheaper explanation exists. Node-perturbation variance scales with the number of perturbed units (width), not the number of weights (width²). See Werfel 2003 and Hiratani 2022 in landscape.md. [unverified as the cause]
- "Integrates well with Adam" (Fig 3): this result is a fitted limit, see claim 1d.

## Gap ledger (decisions the paper or code leaves open)
| Gap | Default we'd use | Risky? |
|---|---|---|
| Baseline LR schedule (constant only) and epochs (1 only) | as in the repo, plus a matched-compute variant | **yes**: the baseline is step-starved at 61 steps |
| What "exceed" means (fitted limit vs measured) | report measured numbers only | **yes** |
| σ per site (0.2, or 0.4 for the deep attention outputs); head σ 0.05 | the repo's values | medium |
| γ = 0 for direct sites, which truncates credit flowing to later tokens through attention | the repo's value | medium: the K→∞ limit is *not* the backprop gradient |
| Draw allocation per layer (hand-tuned lists) | the repo's tuned recipes | medium |
| Reward centring per chunk; none when nrep = 1 | the repo's choice | low |
| Seeds | 42 plus 2 more | low |
| Hardware: the repo states Linux and multi-GPU | single GPU, K ∈ {256, 1024}, Windows | **yes**: memory on a 6GB card is unmeasured |

## The paper's own weak spots
1. **Compute.** Every comparison is at equal tokens. At K = 16k, a step costs about 5,000× a backprop step [unverified estimate: ~16k forward passes vs ~3]. The baseline gets 61 SGD steps at 1M tokens. The authors say they do "not attempt to make it compute-efficient" [source:page].
2. **The headline "exceeds" claims (1c, 1d) are extrapolated limits.** The 20M-token ladder is "still falling" at 16k, with a wide CI.
3. **What Dust's limit actually is.** As K→∞, Dust's estimate converges to the gradient of a *Gaussian-smoothed* loss at each site (Stein's lemma), with a γ = 0 truncation of same-token credit for the direct sites. That is a different objective from backprop's. The "higher-order terms" claim is exactly this smoothing, and it is untested against a backprop control on the same smoothed objective.
4. **Non-differentiable architectures** are the stated motivation, but none are tested [source:page].
5. **Loss is only around 5 nats** (Malladi's critique) [source: digg aggregate].
