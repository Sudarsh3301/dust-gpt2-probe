"""Overnight queue (notes/probes.md): does same-token backprop, the gradient Dust's estimator converges to
(apart from smoothing), train GPT-2 better or worse than plain backprop at equal compute?

  python night.py            run or resume the queue: results/night/*.json, results/night.log
  python night.py --dry-run  every phase for 2 steps (1 rate per arm, 1 seed, 1 batch): results/night_dry/
"""
import argparse
import json
import time
import traceback
from types import SimpleNamespace

import torch

from run import (DATA, DEV, RESULTS, T, accumulate, backprop_dir, boot_ci, cosine, line_search, load_gpt2,
                 seq_losses, set_diag, site_params)

ARMS = [(g, o) for o in ('sgd', 'adamw') for g in ('backprop', 'same_token')]
GRID = dict(sgd=[0.01, 0.03, 0.1, 0.3], adamw=[1e-5, 3e-5, 1e-4, 3e-4])
BATCH = 16          # sequences per step (16,384 tokens)
TRAIN_ROWS = 4800   # pool rows 0-4799 train; 4800-4899 are the 5 replication batches
NO_JITTER = SimpleNamespace(jitter={}, out={})  # backprop_dir without Dust hooks


def night_tokens():
    """Reproduce run.tokens() exactly, then take 4,900 more sequences from later documents."""
    f = DATA / 'tokens_night.pt'
    if f.exists():
        return torch.load(f, weights_only=True)
    import pyarrow.parquet as pq
    import tiktoken
    enc = tiktoken.get_encoding('gpt2')
    docs = iter(pq.read_table(DATA / 'owt_0000.parquet', columns=['text']).column('text'))

    def take(n):
        out = []
        while len(out) < n * (T + 1):
            out += enc.encode_ordinary(next(docs).as_py()) + [enc.eot_token]
        return torch.tensor(out[:n * (T + 1)], dtype=torch.int32).view(n, T + 1)

    val, train, pool = take(256), take(5000), take(4900)
    ref = torch.load(DATA / 'tokens.pt', weights_only=True)
    assert torch.equal(val, ref['val']) and torch.equal(train, ref['train']), 'tokenization does not match data/tokens.pt'
    torch.save(pool, f)
    return pool


def grad_cosine(m, r):
    """Cosine between plain and same-token backprop gradients on one micro-batch, at the current weights."""
    idx, tgt = r[:, :-1], r[:, 1:]
    set_diag(m, False)
    a = backprop_dir(m, NO_JITTER, idx, tgt)
    set_diag(m, True)
    b = backprop_dir(m, NO_JITTER, idx, tgt)
    set_diag(m, False)
    return cosine(a, b)


def train(grad, opt_name, lr, seed, steps, pool, eval_rows, micro, eval_every, cos_rows, path):
    """Continued pretraining of block Linear layers from the warm checkpoint; writes partial results to `path`."""
    m = load_gpt2(DATA / 'warm.pt').requires_grad_(False)
    params = [t for wb in site_params(m).values() for t in wb]
    if opt_name == 'sgd':
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    else:
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    order = torch.randperm(TRAIN_ROWS, generator=torch.Generator().manual_seed(seed))
    log, recent, start = [], [], time.perf_counter()
    for step in range(steps):
        for p in params:
            p.requires_grad_(True)
        set_diag(m, grad == 'same_token')
        for g in opt.param_groups:
            g['lr'] = lr * min(1.0, (step + 1) / 10)
        rows, total = pool[order[step * BATCH:(step + 1) * BATCH]], 0.0
        for j in range(0, BATCH, micro):
            r = rows[j:j + micro].to(DEV).long()
            loss = m.losses(r[:, :-1], r[:, 1:]).mean() * micro / BATCH
            loss.backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        recent.append(total)
        if (step + 1) % eval_every == 0 or step + 1 == steps:
            set_diag(m, False)
            entry = dict(step=step + 1, train_loss=sum(recent) / len(recent),
                         held_out=seq_losses(m, eval_rows, 8).mean().item(),
                         minutes=(time.perf_counter() - start) / 60)
            if cos_rows is not None:
                entry['cos_same_token_vs_backprop'] = grad_cosine(m, cos_rows.to(DEV).long())
            log.append(entry)
            recent = []
            path.write_text(json.dumps(dict(done=False, log=log)))
    set_diag(m, False)
    per_seq = seq_losses(m, eval_rows, 8)
    return dict(grad=grad, opt=opt_name, lr=lr, seed=seed, steps=steps, log=log,
                final_held_out=per_seq.mean().item(), per_seq=per_seq.tolist())


def replicate(pool, b, micro):
    """Probe 2's direction analysis on replication batch b at the warm checkpoint."""
    m = load_gpt2(DATA / 'warm.pt').requires_grad_(False)
    rows = pool[TRAIN_ROWS + BATCH * b:TRAIN_ROWS + BATCH * (b + 1)]
    bp = same = None
    for j in range(0, BATCH, micro):
        r = rows[j:j + micro].to(DEV).long()
        w = micro / BATCH
        set_diag(m, False)
        bp = accumulate(bp, backprop_dir(m, NO_JITTER, r[:, :-1], r[:, 1:]), w)
        set_diag(m, True)
        same = accumulate(same, backprop_dir(m, NO_JITTER, r[:, :-1], r[:, 1:]), w)
    set_diag(m, False)
    etas = torch.logspace(-2, 1, 10).tolist()
    base = seq_losses(m, rows, 8)
    out = dict(batch=b, cos_same_token_vs_backprop=cosine(same, bp), etas=etas)
    for name, d in (('backprop', bp), ('same_token', same)):
        c = line_search(m, d, rows, etas, 8)
        best = int(c.mean(1).argmin())
        out[name] = dict(best_eta=etas[best], delta=boot_ci(c[best] - base), curve=(c.mean(1) - base.mean()).tolist())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dry = args.dry_run
    outdir = RESULTS / ('night_dry' if dry else 'night')
    outdir.mkdir(parents=True, exist_ok=True)
    logfile = RESULTS / ('night_dry.log' if dry else 'night.log')
    tune_steps, main_steps, every = (2, 2, 1) if dry else (100, 300, 50)
    grid = {o: v[:1] for o, v in GRID.items()} if dry else GRID
    seeds, batches = ((0,), (0,)) if dry else ((0, 1, 2), range(5))

    def log(msg):
        line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
        print(line, flush=True)
        with logfile.open('a') as fh:
            fh.write(line + '\n')

    def job(name, fn):
        """Run fn(micro) once; skip if already done; retry with micro 1 on CUDA OOM; log and continue on errors."""
        f = outdir / f'{name}.json'
        if f.exists() and json.loads(f.read_text()).get('done'):
            return json.loads(f.read_text())
        log(f'start {name}')
        for micro in (2, 1):
            try:
                res = fn(micro, f)
                res.update(done=True, micro=micro)
                f.write_text(json.dumps(res))
                log(f'done {name}: ' + (f"held-out {res['final_held_out']:.4f}" if 'final_held_out' in res else 'ok'))
                return res
            except torch.cuda.OutOfMemoryError:
                log(f'{name}: out of memory at micro-batch {micro}')
                torch.cuda.empty_cache()
            except Exception:
                log(f'{name} failed:\n{traceback.format_exc()}')
                break
        f.write_text(json.dumps(dict(done=False, failed=True)))
        return None

    pool = night_tokens()
    from run import tokens
    val = tokens()['val']
    summary = dict(dry_run=dry, tune_steps=tune_steps, main_steps=main_steps, grid=grid)

    # Phase 1: learning-rate sweep, chosen on held-out rows 0-31.
    chosen = {}
    for g, o in ARMS:
        scores = {}
        for lr in grid[o]:
            res = job(f'tune_{g}_{o}_{lr:g}', lambda micro, f, g=g, o=o, lr=lr:
                      train(g, o, lr, 0, tune_steps, pool, val[:32], micro, tune_steps, None, f))
            if res:
                scores[lr] = res['final_held_out']
        if scores:
            best = min(scores, key=scores.get)
            chosen[f'{g}_{o}'] = dict(lr=best, lr_at_edge=best in (grid[o][0], grid[o][-1]) and len(grid[o]) > 1,
                                      tune_held_out=scores)
    summary['chosen'] = chosen
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f'phase 1 done: {json.dumps({k: v["lr"] for k, v in chosen.items()})}')

    # Phase 2: main runs, reported on held-out rows 32-255.
    main = {}
    for g, o in ARMS:
        if f'{g}_{o}' not in chosen:
            continue
        lr = chosen[f'{g}_{o}']['lr']
        for s in seeds:
            res = job(f'main_{g}_{o}_{lr:g}_s{s}', lambda micro, f, g=g, o=o, lr=lr, s=s:
                      train(g, o, lr, s, main_steps, pool, val[32:], micro, every, pool[:2], f))
            if res:
                main.setdefault(f'{g}_{o}', {})[s] = res

    def curves(runs):
        steps = [e['step'] for e in next(iter(runs.values()))['log']]
        held = torch.tensor([[e['held_out'] for e in r['log']] for r in runs.values()])
        return dict(step=steps, mean=held.mean(0).tolist(), min=held.min(0).values.tolist(),
                    max=held.max(0).values.tolist())

    summary['curves'] = {arm: curves(runs) for arm, runs in main.items()}
    summary['same_token_minus_backprop'] = {}
    for o in ('sgd', 'adamw'):
        a, b = main.get(f'same_token_{o}', {}), main.get(f'backprop_{o}', {})
        both = sorted(set(a) & set(b))
        if not both:
            continue
        diff = torch.tensor([a[s]['per_seq'] for s in both]).mean(0) - torch.tensor([b[s]['per_seq'] for s in both]).mean(0)
        summary['same_token_minus_backprop'][o] = dict(
            seeds=both, final_held_out_ci=boot_ci(diff),
            per_seed=[a[s]['final_held_out'] - b[s]['final_held_out'] for s in both])
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log(f"phase 2 done: {json.dumps(summary['same_token_minus_backprop'])}")

    # Phase 3: replication of probe 2's direction analysis on fresh batches.
    reps = [r for b in batches if (r := job(f'replicate_b{b}', lambda micro, f, b=b: replicate(pool, b, micro)))]
    if reps:
        cos = torch.tensor([r['cos_same_token_vs_backprop'] for r in reps])
        drop = {n: torch.tensor([r[n]['delta'][0] for r in reps]) for n in ('backprop', 'same_token')}
        summary['replication'] = dict(
            batches=len(reps), cos_mean=cos.mean().item(), cos_range=[cos.min().item(), cos.max().item()],
            in_sample_drop={n: dict(mean=v.mean().item(), range=[v.min().item(), v.max().item()]) for n, v in drop.items()})
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2))
    log('all phases done')


if __name__ == '__main__':
    main()
