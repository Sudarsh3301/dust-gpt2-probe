# Probes

> **Correction (2026-10-07 13:10).** Our "same-token" gradient detaches keys and values off the diagonal, so W_K and W_V get no credit from later tokens. Dust gives the attention internals (q, k, v, value-embedding gate, value embeddings) credit from later tokens' attention-output errors, decayed by γ^lag with γ = 0.98 [source: dust.py `local_errors`]. Only the direct sites (output projections, MLPs, embeddings) use γ = 0.
> - **Our same-token gradient is therefore more truncated than Dust's real limit.** Below, "full same-token truncation" replaces "Dust's limit" for the overnight result.
> - **Probe 2 stays internally consistent.** Our Dust port also treated q/k/v as same-token sites, so there "limit" means the limit of *our port*, not of official Dust.
> - The whole-network cosine of 0.52 is probably a lower bound for the cosine between official Dust's limit and backprop [unverified].

## Probe 1: line search from a backprop plateau (2026-10-06 23:50, stopped after 1 of 6 batches)
Setup:
- Checkpoint: warm gpt2. Held-out loss went from 3.088 to 2.974, and changed by 0.001 over the last 20% of steps [measured: results/warm.json].
- Directions came from one batch of 2×1024 tokens. The step size was picked on held-out rows 0–31; the loss change was reported on rows 32–95.

Results [measured: results/probe.json, run 0]:
- **No direction lowers held-out loss beyond noise, including exact backprop.**
  - Backprop: Δ = +0.00008 (CI −0.00056 to +0.00072).
  - Every Dust and smoothed direction: Δ between −0.0005 and +0.0001, and every CI crosses 0.
  - Larger steps only raise the loss.
  - At a plateau, the gradient from one 2k-token batch carries no held-out signal, so this design cannot tell the methods apart.
- **Dust's direction is close to orthogonal to backprop's.** Cosine is 0.002 at 216 draws and 0.0001 at 1008 draws, with either σ setting. With 85M parameters and 1k draws, the Dust estimate here behaves like a random direction.
- **Smoothed backprop (noise at all sites at once) is also nearly orthogonal to backprop:** cosine 0.04 with fixed σ, 0.02 with scaled σ. Jittering all 72 sites together distorts the function too much. That makes it a poor stand-in for Dust's limit, since Dust jitters one group at a time.
- **Bug:** the self-cosine printed 1.02–1.03. It is a float32 sum over 85M elements. Fix: compute the cosine in float64. Values near 0 are not materially affected.

Cost: about 9.5 GPU-min. Budget used so far: about 24 of 90 GPU-min.

## Probe 2 smoke test (2026-10-07 00:25; 4k tokens, 8 draws per group; checks the code, not a result)
- Same-token attention matches brute-force per-token gradients to 4e-16 (test_diag.py) [measured].
- The stage runs end to end at nrep 8 on the 6 GB card in 4 min [measured].
- Early sign, too small to cite: same-token backprop vs plain backprop cosine = 0.53 on 4k tokens. If this holds at 16k tokens, Dust's infinite-draw limit is about 58° from backprop before any smoothing.

## Probe 2 (2026-10-07 00:28–01:03, 34.8 min; warm checkpoint, one batch of 16,384 tokens; σ = 0.2 × site RMS)
All numbers are [measured] from results/probe2.json.

Full-network directions (Dust: 1,152 draws, 32 per group):
- Cosine with backprop: Dust −0.010; random −0.0002. Cosine of Dust with same-token backprop: 0.015.
- Same-token backprop vs backprop: cosine **0.531**.
- In-sample line search (best step, 95% CI over 16 sequences):

  | Direction | Loss drop | CI |
  |---|---|---|
  | backprop | −0.061 | −0.068 to −0.053 |
  | **same-token backprop** | **−0.133** | −0.145 to −0.121 |
  | Dust | 0.000 | −0.0014 to +0.0013 |
  | random | −0.0002 | — |

- Held-out line search: no direction beats noise. Every CI crosses 0, as in probe 1.

Draw-scaling curves (cosine at 32 to 1,024 draws on one group, 16k tokens; references are exact, with 16 smoothing samples):

| Group | vs backprop @1024 | vs Dust's limit @1024 | cos(backprop, limit) | Ratio of the first two | What separates the limit from backprop |
|---|---|---|---|---|---|
| layer 0 MLP in | 0.111 | 0.279 | 0.359 | 0.40 | both same-token credit (0.49) and smoothing (0.20) |
| layer 5 writers | 0.233 | 0.421 | 0.572 | 0.55 | same-token credit only (smoothed vs backprop 0.995) |
| layer 11 MLP in | 0.453 | 0.548 | 0.778 | 0.83 | smoothing only (same-token vs backprop 0.999) |

- On every group, the ratio of the two cosines tracks cos(backprop, limit). That is what you expect if Dust converges to the same-token + smoothed gradient and not to backprop.
- Fitted c_max vs the limit: 0.95 (layer 0) and 1.07 (layer 5). Layer 11 gives 0.70, likely held down by Monte Carlo noise in the 16-sample reference. Vs backprop, the fits sit well below 1.
- **Fits are loosely constrained**: the measured cosines reach only 0.28–0.55. Draws needed for half of c_max, per group: about 330 (layer 11), 1,670 (layer 5) and 3,700 (layer 0). Over 36 groups, that implies roughly 12k–130k draws per update for a useful direction on GPT-2 124M [extrapolated].

Caveats: one batch, one checkpoint, one σ; smoothed references come from 16 noise samples; the same-token trick is verified against brute force only on a tiny model (test_diag.py).

## Overnight run: training with full same-token truncation (2026-10-07 02:21–06:11, 34 runs, 0 failed)
All numbers are [measured]: results/night/summary.json and results/night.log. Warm checkpoint on held-out rows 32–255: 2.9810.

| Arm (rate chosen) | Final held-out, mean of 3 seeds | Gain vs warm |
|---|---|---|
| backprop · SGD (0.01, **edge**) | 2.9796 | 0.0014 |
| same-token · SGD (0.03) | 2.9800 | 0.0010 |
| backprop · AdamW (3e-5) | 2.9767 | 0.0043 |
| same-token · AdamW (1e-5, **edge**) | 2.9785 | 0.0025 |

- **Same-token minus backprop**, paired bootstrap over 224 sequences:
  - SGD: +0.00047 (CI +0.00002 to +0.00091); per seed +0.00075, +0.00075, −0.00008.
  - AdamW: +0.00178 (CI +0.00114 to +0.00247); per seed +0.0016, +0.0021, +0.0016.
- Same-token training gets about 70% (SGD) and about 58% (AdamW) of backprop's held-out gain at equal compute. **Full same-token truncation (no cross-token credit anywhere, including keys and values) is worse than backprop here**, and clearly so with AdamW. This truncation is harsher than official Dust's limit, so it overstates Dust's handicap (see the correction at the top).
- **Replication** (5 new 16k-token batches): cosine(same-token, backprop) = 0.517 (range 0.510–0.533). One-step in-sample loss drop: same-token 0.142 (0.124–0.162) vs backprop 0.071 (0.058–0.082). The 2× in-sample advantage holds on all 5 batches, but it **does not carry over** to held-out loss during training.
- **Caveats:**
  - Two arms chose the smallest rate in their grid (backprop · SGD and same-token · AdamW), so the lower edge was not explored for them.
  - The tuning gaps are tiny (for example 2.9261 vs 2.9259) and measured on only 32 sequences.
  - Starting from a plateau, every arm gains only 0.001–0.004 nats over 4.9M tokens.
  - Only block Linear layers train, and no arm uses smoothing.
  - One model size (124M), 300 steps.

## Probe 1 conclusion
What it means: the line search at a plateau has no statistical power. To answer the question we need three things:
1. a much larger batch for each direction, plus a random-direction control and a same-batch (in-sample) line search;
2. a measurement of how many draws Dust needs on GPT-2 before its direction carries any signal;
3. a per-group smoothed control, which is the true limit of Dust's estimate.
