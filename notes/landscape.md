# Landscape (searched 2026-10-06; the paper is about 1 day old)

Grades: [source] means the page was opened. [unverified] means a snippet only.

## Code and its scars
- Official repo: https://github.com/qlabs-eng/dust, MIT, 4 commits, 0 issues, 0 forks [source]. The code matches the paper's description of direct draws (γ = 0) and attention-internal credit (γ ≈ 1) [source: code and page].
- Reimplementations: none found on GitHub or Hugging Face [source: searches].

## Reception
- Hacker News, 158 points (https://news.ycombinator.com/item?id=49970871): comments call it a "Monte Carlo gradient estimate", raise gradient-free complexity gaps, and dispute the parallelism claim [source: Algolia].
- Malladi (MeZO author) on X: a loss around 5 is "super high"; the x-axis should be steps or GPU-hours, not tokens [source: https://digg.com/ai/5zv01jcm].
- byteiota makes the same tokens-vs-GPU-hours point [source: https://byteiota.com/backprop-has-a-real-challenger-dust-trains-transformers/].

## Prior art for the mechanism
- Node perturbation: Werfel, Xie & Seung 2003 [unverified]; Widrow & Lehr 1990 (cited by Dust).
- Activity perturbation has lower variance than weight perturbation: Ren et al. 2023, https://arxiv.org/abs/2210.03310 [source].
- Flipout gives per-example independent perturbations with linear variance reduction: https://arxiv.org/abs/1803.04386 [source].
- Node perturbation over time in RNNs: https://arxiv.org/abs/2405.08967 [source].
- EGGROLL, low-rank weight-space ES: https://arxiv.org/abs/2511.16652 [source].
- Cost of avoiding backprop: at matched compute, zeroth-order lags far behind (OPT-13B 77% vs 94%), https://arxiv.org/abs/2506.21833 [source].
- Zeroth-order is implicitly biased toward flat minima: https://arxiv.org/abs/2506.05454 [source].
- ES optimizes robustness to perturbation, which is more than finite differences: Lehman et al. 2018, https://arxiv.org/abs/1712.06568 [source].
- Mollifying networks (backprop on a noise-smoothed objective): https://arxiv.org/abs/1608.04980 [source].
- STE plus zeroth-order hybrid for non-differentiable ops (FOGZO): https://arxiv.org/abs/2510.23926 [source].
- Bigger nets train faster with perturbative training: https://arxiv.org/abs/2501.15403 [source]. Node-perturbation training time is nearly width-independent (Hiratani et al., NeurIPS 2022) [source].

## Already done (kills or reshapes directions)
- Bigger-is-better at fixed population: shown at small scale already (2501.15403, and Dust itself). Dropped.
- Tokens as a virtual population: the idea is Flipout plus Ren et al. Dropped as a novelty claim.
- Same-token credit: Dust already uses γ for the attention internals, so this shrinks to an ablation.

## Open complaints (raised, but nobody has tested them)
- Tokens vs compute or GPU-hours: raised by Malladi, HN and byteiota. **No one has run backprop at matched FLOPs over the same tokens.**
- "Monte Carlo gradient", "it's just smoothing": raised in passing. **No one has run the backprop control on the same Gaussian-smoothed objective.**

## Independent reproduction: huskydoge (2026-10-07) [source: report + repo b452f0f, read]
- Links: https://huskydoge.github.io/dust/ and https://github.com/huskydoge/dust. It is a fork of the official repo that adds an AdamW backprop baseline, Dust under Adam, and continued pretraining (`dust_cpt.py`).
- Paper numbers reproduce with SGD. For example, 1M tokens with Dust at population 1k scores 5.933 vs 5.934 in the paper.
- **With AdamW, backprop is far better:**
  - 1M tokens: backprop 5.364 vs Dust-Adam at 16k 5.382, and Dust-Adam at 256 is 5.843.
  - 10M tokens: backprop 3.939 vs Dust-Adam at 16k 4.436, about 0.5 nat behind.
  - At 10M tokens, AdamW backprop beats SGD backprop by 1.05 nat.
- **Cost:** Dust at 16k uses 478–2,316× backprop's GPU-hours.
- **Continued pretraining with actual Dust** on their 37.7M model with BPE-4096 (checkpoints at 3.980 and 3.254):
  - At 0.0005, Dust gets 65–85% of backprop's loss drop over 200 steps.
  - At the from-scratch learning rates, Dust does not improve either checkpoint.
  - The report notes that "the proposal asked for a checkpoint below a loss of 3 with the GPT-2 tokenizer", and credits the idea to You Jiacheng. Their checkpoints start higher.
- **Truncation, as a hypothesis only.** They write that "our reading, not the paper's: with γ=0 the estimate … targets a truncated gradient rather than backprop's". They neither compute that truncated gradient nor train with it. Their cosines are the paper's fitted values put through the paper's own fits, not new measurements.
- **Overlap with this repo:**
  - The continued-pretraining-with-Dust idea is done, on a smaller model.
  - **Not done:** the exact same-token limit, measured convergence to it, and training with it.
  - Our result, that full same-token truncation (noise-free) gets 58–70% of backprop's gain on GPT-2 124M, partly fills the gap their hypothesis leaves. It is harsher than Dust's real truncation, which keeps γ-decayed future credit for keys and values (see probes.md, correction). The setups also differ, so comparing the percentages across studies is [unverified].

## Freshest related work (scoop risk)
- arXiv 2609.37899, SOMA (Chaubard, Kochenderfer, Ré, Stanford, 2026-09-29) [source: PDF read, pp. 1–12 plus appendices B, I, O].
  - What it is: weight-space SPSA on byte-level LSTM experts (8.44M parameters total), sharded into N independently trained experts. Each expert gets its own loss, which cuts the relative gradient variance to about 1/N.
  - Axis: GPU-hours, not tokens.
  - Not activation or node perturbation; no transformers; does not cite Dust.
  - **Appendix Table 5:** backpropagation-through-time (BPTT) beats every ZO recipe at an equal 50–100 GPU-hours, reaching about 1.02–1.18 nats/byte vs 1.83–2.12 for SOMA. In their words: "These SOMA recipes do not show a training advantage over backpropagation." So matched compute has already been shown to sink ZO pretraining, for SPSA on LSTMs.
  - Overlap with our candidates:
    - Matched-compute backprop: reduced novelty. The general result is already out; only the Dust/transformer data-limited version remains.
    - Smoothing control: none.
    - Non-differentiable port: none. But it cites Yang & Aamodt, NeurIPS 2025, "Improving the straight-through estimator with zeroth-order information" (FOGZO), which is nearest prior work for direction 3.
  - Conceptual link: SOMA's "independent losses reduce variance" parallels Dust jittering different layer types separately to reduce interference.
- Dust has no arXiv version and no citing papers yet.
