# dust-gpt2-probe

A small probe of one argument for [Dust](https://qlabs.sh/research/dust), a method that trains transformers without backpropagation: that zeroth-order search can find descent directions which backprop misses.

**Question.** Start from a GPT-2 124M checkpoint that backprop has already trained on OpenWebText. Does Dust's update direction lower held-out loss where backprop's direction does not?

**Scope.** This is a weekend-scale probe on one model size and one checkpoint, on a 6 GB laptop GPU. The Dust code here is my own port of the idea, not the official release. Read the [limits](#limits) before citing any number.

## What I did

1. Rewrote GPT-2 124M in a nanoGPT style with separate q, k and v layers, so each can take a perturbation hook. The loss matches Hugging Face (see "Port check" below).
2. Trained the checkpoint with AdamW on about 4M OpenWebText tokens. The warm-up run took validation loss from 3.088 to 2.974 (`results/warm.json`).
3. From that checkpoint, built four update directions on one 16,384-token batch: exact backprop, a "same-token" backprop gradient, a Dust estimate, and a random direction. Then ran a line search along each one.
4. Trained 300 steps with plain backprop and with the same-token gradient, under SGD and AdamW, 3 seeds each.

The **same-token gradient** is backprop where each token's loss reaches only that token's own path. It is the gradient that Dust's direct draws target (credit from the same token only). `test_diag.py` checks it against a brute-force per-token backward pass on a tiny CPU model.

## Results

All tables below are printed by `make_tables.py` from the JSON files in `results/`.

Port check: our loss 2.827711, Hugging Face 2.827712, difference 7.2e-07

### Line search along each direction (one batch of 16,384 tokens; Dust uses 1152 draws)

Loss change at the best step size, with a 95% bootstrap CI over sequences. Negative is better.

| Direction | In-sample | Held-out |
|---|---|---|
| Backprop | -0.0605 [-0.0678, -0.0526] | +0.0002 [-0.0006, +0.0009] |
| Same-token backprop | -0.1327 [-0.1449, -0.1205] | -0.0001 [-0.0008, +0.0005] |
| Dust (this port) | -0.0000 [-0.0014, +0.0013] | -0.0005 [-0.0012, +0.0001] |
| Random | -0.0002 [-0.0018, +0.0014] | -0.0006 [-0.0012, -0.0000] |

### Cosine between directions

| Pair | Cosine |
|---|---|
| dust vs backprop | -0.0097 |
| dust vs backprop_same_token | +0.0154 |
| backprop_same_token vs backprop | +0.5306 |
| random vs backprop | -0.0002 |

### Continued training from the warm checkpoint (300 steps, 3 seeds)

| Arm | Learning rate | Final held-out loss (mean of seeds) | Rate at grid edge |
|---|---|---|---|
| backprop sgd | 0.01 | 2.9796 (n=3) | True |
| same token sgd | 0.03 | 2.9800 (n=3) | False |
| backprop adamw | 3e-05 | 2.9767 (n=3) | False |
| same token adamw | 1e-05 | 2.9785 (n=3) | True |

Same-token minus backprop, final held-out loss, paired bootstrap over sequences (positive means same-token is worse):

- sgd: +0.0005 [+0.0000, +0.0009]; per seed +0.0008, +0.0007, -0.0001
- adamw: +0.0018 [+0.0011, +0.0025]; per seed +0.0016, +0.0021, +0.0016

### Replication on 5 new batches

- Cosine, same-token vs backprop: 0.517 (range 0.510 to 0.533)
- In-sample one-step loss change, backprop: -0.0711 (range -0.0822 to -0.0581)
- In-sample one-step loss change, same-token: -0.1423 (range -0.1621 to -0.1244)

### What the tables say

- **Dust at about 1k draws behaved like a random direction.** Its loss change is indistinguishable from the random control, in-sample and held-out, and its cosine with backprop is near zero.
- **The same-token gradient roughly doubles the in-sample loss drop** compared with exact backprop. This held on all 5 replication batches.
- **No direction gave a held-out gain.** Every held-out change is within about 0.001 nat of zero, including exact backprop and the random control, so none counts as a gain. One batch at this checkpoint has little power to separate methods.
- **In training, the in-sample advantage did not carry over.** Same-token training ended slightly behind backprop. The gap is small (a few thousandths of a nat) but has the same sign on every AdamW seed.

## Limits

- **The Dust here is a port.** It treats q, k and v as direct draw sites with no future-token credit. Official Dust gives keys, values and gates credit from later tokens. So "Dust" in the line-search table means this port.
- **The same-token gradient is a harsher truncation than real Dust.** It gives keys and values no later-token credit at all. It therefore overstates how much Dust loses against backprop.
- **Draw counts are far below the paper's.** The probe uses about 1k draws on a 124M model. The paper's main setting uses up to 16,384 draws on a 38M model. In my draw-scaling fits, a useful direction needed roughly 12k to 130k draws per update. That number is an extrapolation from loosely constrained fits (`notes/probes.md`).
- **The gains are tiny.** Every training arm improved held-out loss by only a few thousandths of a nat over 300 steps. Two arms picked the lowest learning rate in their grid, so the lower edge was not explored. Tuning used 32 held-out sequences.
- **One checkpoint, one model size, mostly one batch for the line search.** Only block Linear layers train; embeddings, LayerNorm and the head are frozen for every method.

## Reproduce

Needs Python 3.11, a CUDA GPU with about 6 GB of memory, and the packages in `requirements.txt`.

```bash
pip install -r requirements.txt
mkdir -p data/gpt2
# GPT-2 weights and config: https://huggingface.co/openai-community/gpt2
#   -> data/gpt2/model.safetensors, data/gpt2/config.json
# First OpenWebText shard (303 MB), from the parquet export of Skylion007/openwebtext:
#   plain_text/train/0000.parquet on branch refs/convert/parquet -> data/owt_0000.parquet

python test_diag.py             # CPU, a few seconds: same-token gradient vs brute force
python run.py --stage bench     # port check and timing
python run.py --stage warm      # make the warm checkpoint (about 12 min on an RTX 3050)
python run.py --stage probe2    # directions, line search, draw-scaling curves
python night.py                 # training comparison; resumable
python make_tables.py           # print the tables above from results/
```

`python night.py --dry-run` runs every phase for 2 steps as a smoke test.

## Layout

| Path | Contents |
|---|---|
| `run.py` | Port, warm-up, directions, line search (stages `bench`, `warm`, `probe`, `probe2`) |
| `night.py` | Training comparison queue |
| `test_diag.py` | Same-token gradient check |
| `make_tables.py` | Prints the result tables |
| `results/` | Raw JSON and logs written by the scripts |
| `notes/` | `probes.md` (findings and corrections), `landscape.md` (related work), `distill.md` (paper summary) |

## Credits

- Dust: Dahal, Mandal, Gülbahar and Vegesna, Q Labs, October 2026 ([paper](https://qlabs.sh/research/dust), [official code](https://github.com/qlabs-eng/dust), MIT). This repo does not contain their code.
- Related independent reproduction with an AdamW baseline: [huskydoge/dust](https://github.com/huskydoge/dust).
- GPT-2: OpenAI. Data: OpenWebText.

## License

MIT. See `LICENSE`.
