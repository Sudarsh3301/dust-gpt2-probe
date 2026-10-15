"""Print the README result tables from results/*.json. Usage: python make_tables.py [results_dir]"""
import json
import sys
from pathlib import Path

R = Path(sys.argv[1] if len(sys.argv) > 1 else 'results')


def load(p):
    return json.loads((R / p).read_text())


def ci(d):  # [point, lo, hi]
    return f'{d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]'


bench, p2, night = load('bench.json'), load('probe2.json'), load('night/summary.json')

print(f"Port check: our loss {bench['port_loss_ours']:.6f}, Hugging Face {bench['port_loss_hf']:.6f}, "
      f"difference {abs(bench['port_loss_ours'] - bench['port_loss_hf']):.1e}\n")

print(f"### Line search along each direction (one batch of {p2['tokens']:,} tokens; Dust uses {p2['dust_draws']} draws)\n")
print('Loss change at the best step size, with a 95% bootstrap CI over sequences. Negative is better.\n')
print('| Direction | In-sample | Held-out |\n|---|---|---|')
labels = dict(backprop='Backprop', backprop_same_token='Same-token backprop', dust='Dust (this port)', random='Random')
for k, label in labels.items():
    ls = p2['line_search'][k]
    print(f"| {label} | {ci(ls['in_sample']['delta'])} | {ci(ls['held_out']['delta'])} |")

c = p2['cosine']
print('\n### Cosine between directions\n')
print('| Pair | Cosine |\n|---|---|')
for k in ('dust vs backprop', 'dust vs backprop_same_token', 'backprop_same_token vs backprop', 'random vs backprop'):
    print(f'| {k} | {c[k]:+.4f} |')

print('\n### Continued training from the warm checkpoint (300 steps, 3 seeds)\n')
print('| Arm | Learning rate | Final held-out loss (mean of seeds) | Rate at grid edge |\n|---|---|---|---|')
for arm, info in night['chosen'].items():
    runs = [json.loads(f.read_text())['final_held_out'] for f in sorted((R / 'night').glob(f'main_{arm}_*_s*.json'))]
    print(f"| {arm.replace('_', ' ')} | {info['lr']:g} | {sum(runs) / len(runs):.4f} (n={len(runs)}) | {info['lr_at_edge']} |")

print('\nSame-token minus backprop, final held-out loss, paired bootstrap over sequences (positive means same-token is worse):\n')
for opt, d in night['same_token_minus_backprop'].items():
    print(f"- {opt}: {ci(d['final_held_out_ci'])}; per seed {', '.join(f'{x:+.4f}' for x in d['per_seed'])}")

r = night['replication']
print(f"\n### Replication on {r['batches']} new batches\n")
print(f"- Cosine, same-token vs backprop: {r['cos_mean']:.3f} (range {r['cos_range'][0]:.3f} to {r['cos_range'][1]:.3f})")
for k in ('backprop', 'same_token'):
    d = r['in_sample_drop'][k]
    print(f"- In-sample one-step loss change, {k.replace('_', '-')}: {d['mean']:+.4f} (range {d['range'][0]:+.4f} to {d['range'][1]:+.4f})")
