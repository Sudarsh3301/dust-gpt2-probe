"""Does zeroth-order search (Dust) find descent that backprop misses, starting from a backprop-trained GPT-2?

Stages:
  bench  verify the GPT-2 port against Hugging Face, measure speed and memory
  warm   backprop on OpenWebText until the loss plateaus (makes the starting checkpoint)
  probe  from that checkpoint, compare update directions (backprop, smoothed backprop, Dust)
         by held-out line search and per-layer cosine (first design; underpowered, see notes/probes.md)
  probe2 16k-token directions with a random control, plus Dust draw-scaling curves against
         its exact infinite-draw limit (same-token credit + per-group Gaussian smoothing)
"""
import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).parent
DATA, RESULTS = ROOT / 'data', ROOT / 'results'
T, DEV = 1024, 'cuda'
KINDS = ('q', 'k', 'v', 'attn_proj', 'fc', 'mlp_proj')
# Sites jittered together share one reward, as Dust does for its output projections.
GROUPS = (('q', 'k', 'v'), ('attn_proj', 'mlp_proj'), ('fc',))


class Block(nn.Module):
    def __init__(self, d=768, heads=12):
        super().__init__()
        self.heads, self.diag = heads, False
        self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        # q, k, v are split out of GPT-2's fused c_attn so that each gets its own jitter hook.
        self.q, self.k, self.v, self.attn_proj = (nn.Linear(d, d) for _ in range(4))
        self.fc, self.mlp_proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, L, C = x.shape
        y = self.ln_1(x)
        q, k, v = (m(y).view(B, L, self.heads, C // self.heads).transpose(1, 2) for m in (self.q, self.k, self.v))
        if self.diag:
            # Same-token credit: forward values are unchanged, but gradient reaches a position's k and v
            # only through its own query, so backprop returns sum_t d loss_t / d y_t, the target Dust's
            # same-token reward estimates.
            scale = q.shape[-1] ** -0.5
            own = (q.detach() * k).sum(-1) * scale  # q already gets its full gradient from the term below
            att = q @ k.detach().transpose(-1, -2) * scale + torch.diag_embed(own - own.detach())
            causal = torch.ones(L, L, dtype=torch.bool, device=x.device).tril()
            att = att.masked_fill(~causal, float('-inf')).softmax(-1)
            y = att @ v.detach() + att.diagonal(dim1=-2, dim2=-1).unsqueeze(-1) * (v - v.detach())
            y = y.transpose(1, 2).reshape(B, L, C)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, L, C)
        x = x + self.attn_proj(y)
        return x + self.mlp_proj(F.gelu(self.fc(self.ln_2(x)), approximate='tanh'))


class GPT(nn.Module):
    def __init__(self, layers=12, d=768, vocab=50257):
        super().__init__()
        self.wte, self.wpe = nn.Embedding(vocab, d), nn.Embedding(T, d)
        self.h = nn.ModuleList(Block(d) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)

    def losses(self, idx, tgt, start=0, x=None, cache=None, amp=True):
        """Per-token cross-entropy. Starts at block `start` from residual `x`; `cache` collects each block's input."""
        with torch.autocast('cuda', torch.bfloat16, enabled=amp):
            if x is None:
                x = self.wte(idx) + self.wpe(torch.arange(idx.shape[1], device=idx.device))
            for i in range(start, len(self.h)):
                if cache is not None:
                    cache[i] = x
                x = self.h[i](x)
            x = self.ln_f(x).flatten(0, 1)
        # The head is tied to the token embedding, as in GPT-2. Chunking keeps float32 logits at <= 2048 x 50257,
        # so many draws fit in one batched forward.
        out = []
        for xc, tc in zip(x.split(2048), tgt.flatten().split(2048)):
            with torch.autocast('cuda', torch.bfloat16, enabled=amp):
                logits = F.linear(xc, self.wte.weight)
            out.append(F.cross_entropy(logits.float(), tc, reduction='none'))
        return torch.cat(out).view_as(tgt)


def load_gpt2(state=None):
    from safetensors.torch import load_file
    sd = {k.removeprefix('transformer.'): v for k, v in load_file(DATA / 'gpt2' / 'model.safetensors').items()}
    m = GPT()
    with torch.no_grad():
        m.wte.weight.copy_(sd['wte.weight'])
        m.wpe.weight.copy_(sd['wpe.weight'])
        for n in ('weight', 'bias'):
            getattr(m.ln_f, n).copy_(sd[f'ln_f.{n}'])
        for i, b in enumerate(m.h):
            p = f'h.{i}.'
            for ln in ('ln_1', 'ln_2'):
                for n in ('weight', 'bias'):
                    getattr(getattr(b, ln), n).copy_(sd[f'{p}{ln}.{n}'])
            W, bias = sd[p + 'attn.c_attn.weight'].t(), sd[p + 'attn.c_attn.bias']  # Conv1D stores [in, out]
            for j, mod in enumerate((b.q, b.k, b.v)):
                mod.weight.copy_(W[j * 768:(j + 1) * 768])
                mod.bias.copy_(bias[j * 768:(j + 1) * 768])
            for mod, key in ((b.attn_proj, 'attn.c_proj'), (b.fc, 'mlp.c_fc'), (b.mlp_proj, 'mlp.c_proj')):
                mod.weight.copy_(sd[p + key + '.weight'].t())
                mod.bias.copy_(sd[p + key + '.bias'])
    if state is not None:
        m.load_state_dict(torch.load(state, map_location='cpu', weights_only=True))
    return m.to(DEV)


def tokens():
    """OpenWebText shard 0: the first 256 sequences are held out; training sequences come from later documents."""
    f = DATA / 'tokens.pt'
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

    d = dict(val=take(256), train=take(5000))
    torch.save(d, f)
    return d


@torch.no_grad()
def seq_losses(m, rows, bs=4):
    out = []
    for r in rows.split(bs):
        r = r.to(DEV).long()
        out.append(m.losses(r[:, :-1], r[:, 1:]).mean(1))
    return torch.cat(out).cpu()


def site_params(m):
    return {(i, k): (getattr(b, k).weight, getattr(b, k).bias) for i, b in enumerate(m.h) for k in KINDS}


class Dust:
    """Dust's direct-draw estimator on GPT-2's block Linear outputs.

    ponytail: same-token reward only (gamma = 0) for every site, including q/k/v. Dust's attention-internal
    credit and its head/embedding estimators are not ported; those parameters stay frozen for every method.
    """

    def __init__(self, m):
        self.m, self.jitter, self.inp, self.out, self.capturing = m, {}, {}, {}, False
        for i, b in enumerate(m.h):
            for k in KINDS:
                getattr(b, k).register_forward_hook(self.hook((i, k)))

    def hook(self, key):
        def f(mod, args, out):
            if self.capturing:
                self.inp[key], self.out[key] = args[0].float(), out.float()
            if key in self.jitter:
                return out + self.jitter[key].to(out.dtype)
        return f

    @torch.no_grad()
    def capture(self, idx, tgt, sigma, scaled):
        """Clean pass: caches each site's input and output and returns the per-site noise scale."""
        self.cache, self.idx, self.tgt = {}, idx, tgt
        self.capturing = True
        self.clean = self.m.losses(idx, tgt, cache=self.cache)
        self.capturing = False
        return {key: sigma * (o.pow(2).mean().sqrt().item() if scaled else 1.0) for key, o in self.out.items()}

    @torch.no_grad()
    def group_estimates(self, i, group, draws, sig, nrep, checkpoints):
        """Weight/bias gradient estimates for one jitter group after each draw count in `checkpoints`.

        All nrep draws of a chunk run as one batched forward, rerunning only from block i on.
        """
        keys = [(i, k) for k in group]
        x = self.cache[i].repeat(nrep, 1, 1)
        ti, tt, n = self.idx.repeat(nrep, 1), self.tgt.repeat(nrep, 1), self.tgt.numel()
        err = {key: torch.zeros_like(self.out[key]) for key in keys}
        out = {}
        for c in range(draws // nrep):
            a = {key: torch.randn((nrep,) + self.out[key].shape, device=DEV) for key in keys}
            self.jitter = {key: sig[key] * a[key].flatten(0, 1) for key in keys}
            pert = self.m.losses(ti, tt, start=i, x=x).view(nrep, *self.tgt.shape)
            self.jitter = {}
            r = pert - self.clean
            r = r - r.mean(0, keepdim=True)  # centred across the chunk, as in Dust
            for key in keys:
                err[key] += torch.einsum('nbt,nbtd->btd', r, a[key]) / sig[key]
            done = (c + 1) * nrep
            if done in checkpoints:
                out[done] = {key: (torch.einsum('btd,bti->di', e / done, self.inp[key]) / n, e.sum((0, 1)) / done / n)
                             for key, e in err.items()}
        return out

    @torch.no_grad()
    def estimate(self, idx, tgt, draws, sigma, scaled, nrep=2):
        sig = self.capture(idx, tgt, sigma, scaled)
        per_group = max(nrep, draws // (len(GROUPS) * len(self.m.h)) // nrep * nrep)
        grads = {}
        for i in range(len(self.m.h)):
            for group in GROUPS:
                est = self.group_estimates(i, group, per_group, sig, nrep, {per_group})[per_group]
                grads.update({key: (w.cpu(), b.cpu()) for key, (w, b) in est.items()})
        return grads, per_group


def backprop_dir(m, dust, idx, tgt, sig=None, samples=1):
    """Exact gradient of the mean loss; with `sig`, averaged over Gaussian jitter at every site (smoothed loss).

    ponytail: smoothing jitters all sites at once; Dust jitters one group at a time. Per-group smoothing costs 36x.
    """
    P = site_params(m)
    for W, b in P.values():
        W.requires_grad_(True)
        b.requires_grad_(True)
    acc = {k: (torch.zeros_like(W), torch.zeros_like(b)) for k, (W, b) in P.items()}
    for _ in range(samples):
        if sig:
            dust.jitter = {key: s * torch.randn_like(dust.out[key]) for key, s in sig.items()}
        m.losses(idx, tgt).mean().backward()
        dust.jitter = {}
        for k, (W, b) in P.items():
            acc[k][0].add_(W.grad / samples)
            acc[k][1].add_(b.grad / samples)
            W.grad = b.grad = None
    for W, b in P.values():
        W.requires_grad_(False)
        b.requires_grad_(False)
    return {k: (w.cpu(), b.cpu()) for k, (w, b) in acc.items()}


def flat(d, keys=None):
    return torch.cat([t.flatten().double().cpu() for k in (keys or d) for t in d[k]])  # float64: 85M-term sums


def cosine(a, b, keys=None):
    x, y = flat(a, keys), flat(b, keys)
    return (x @ y / (x.norm() * y.norm())).item()


@torch.no_grad()
def line_search(m, d, rows, etas, bs=4):
    """Per-sequence losses after a step of size eta along the unit-norm direction -d."""
    P = site_params(m)
    base = {k: (W.clone(), b.clone()) for k, (W, b) in P.items()}
    norm = flat(d).norm().item()
    unit = {k: (a.to(DEV) / norm, b.to(DEV) / norm) for k, (a, b) in d.items()}  # one host->GPU copy per direction
    out = []
    for eta in etas:
        for k, (W, b) in P.items():
            torch.add(base[k][0], unit[k][0], alpha=-eta, out=W)
            torch.add(base[k][1], unit[k][1], alpha=-eta, out=b)
        out.append(seq_losses(m, rows, bs))
    for k, (W, b) in P.items():
        W.copy_(base[k][0])
        b.copy_(base[k][1])
    return torch.stack(out)


def boot_ci(x, n=2000):
    g = torch.Generator().manual_seed(0)
    means = x[torch.randint(len(x), (n, len(x)), generator=g)].mean(1)
    return [x.mean().item(), means.quantile(0.025).item(), means.quantile(0.975).item()]


def save(name, obj):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f'{name}.json').write_text(json.dumps(obj, indent=2))


def timed(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / reps


def bench(args):
    from transformers import GPT2LMHeadModel
    d = tokens()
    r = d['val'][:2].to(DEV).long()
    m = load_gpt2().requires_grad_(False)
    with torch.no_grad():
        ours = m.losses(r[:, :-1], r[:, 1:], amp=False).mean().item()
        hf = GPT2LMHeadModel.from_pretrained(str(DATA / 'gpt2')).to(DEV).eval()
        logits = hf(r[:, :-1]).logits
        theirs = F.cross_entropy(logits.flatten(0, 1), r[:, 1:].flatten()).item()
        del hf, logits
    res = dict(port_loss_ours=ours, port_loss_hf=theirs, port_ok=abs(ours - theirs) < 1e-3,
               val_tokens=d['val'].numel(), train_tokens=d['train'].numel())
    torch.cuda.reset_peak_memory_stats()
    for B in (2, 4):
        rb = d['train'][:B].to(DEV).long()
        res[f'forward_s_{B}x1024'] = timed(lambda: torch.no_grad()(m.losses)(rb[:, :-1], rb[:, 1:]))
    dust = Dust(m)
    rb = d['train'][:2].to(DEV).long()
    t = time.perf_counter()
    _, per_group = dust.estimate(rb[:, :-1], rb[:, 1:], 72, 0.2, True)
    torch.cuda.synchronize()
    res['dust_s_per_draw'] = (time.perf_counter() - t) / (per_group * len(GROUPS) * len(m.h))
    res['backprop_s_2x1024'] = timed(lambda: backprop_dir(m, dust, rb[:, :-1], rb[:, 1:]), reps=2)
    res['peak_gb'] = torch.cuda.max_memory_allocated() / 2**30
    save('bench', res)
    print(json.dumps(res, indent=2))


def warm(args):
    """Backprop fine-tune on 4M OpenWebText tokens: closes the WebText->OpenWebText gap so the start is a plateau."""
    d = tokens()
    train = d['train'][:4000][torch.randperm(4000, generator=torch.Generator().manual_seed(0))]
    val = d['val']
    m = load_gpt2()
    opt = torch.optim.AdamW(m.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps, micro = len(train) // 8, 2
    log = [dict(step=0, val=seq_losses(m, val).mean().item())]
    start = time.perf_counter()
    for step in range(steps):
        lr = args.lr * min(1.0, (step + 1) / 50) * 0.5 * (1 + math.cos(math.pi * step / steps))
        for g in opt.param_groups:
            g['lr'] = lr
        for j in range(0, 8, micro):
            r = train[step * 8 + j: step * 8 + j + micro].to(DEV).long()
            (m.losses(r[:, :-1], r[:, 1:]).mean() * micro / 8).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (step + 1) % 50 == 0 or step + 1 == steps:
            log.append(dict(step=step + 1, val=seq_losses(m, val).mean().item(), lr=lr,
                            minutes=(time.perf_counter() - start) / 60))
            print(log[-1], flush=True)
    torch.save(m.state_dict(), DATA / 'warm.pt')
    tail = [e['val'] for e in log if e['step'] >= 0.8 * steps]
    save('warm', dict(lr=args.lr, steps=steps, tokens=steps * 8 * T, log=log,
                      plateau_change_last_20pct=max(tail) - min(tail)))


def probe(args):
    d = tokens()
    rows, pool = d['val'][:96], d['train'][4000:]  # eta picked on the first 32 rows, reported on the other 64
    etas = torch.logspace(-2, 1, 10).tolist()
    start, out = time.perf_counter(), dict(etas=etas, draws=args.draws, sigma=args.sigma, runs=[])
    for ckpt in ('warm', 'gpt2'):
        m = load_gpt2(DATA / 'warm.pt' if ckpt == 'warm' else None).requires_grad_(False)
        dust = Dust(m)
        base = seq_losses(m, rows)
        for bi in range(args.batches):
            if (time.perf_counter() - start) / 60 > args.max_minutes:
                out['stopped_early'] = True
                break
            torch.manual_seed(bi)
            r = pool[2 * bi: 2 * bi + 2].to(DEV).long()
            idx, tgt = r[:, :-1], r[:, 1:]
            dirs = {'backprop': backprop_dir(m, dust, idx, tgt)}
            for scaled in (False, True):
                tag = 'scaled' if scaled else 'fixed'
                sig = dust.capture(idx, tgt, args.sigma, scaled)
                dirs[f'smoothed_{tag}'] = backprop_dir(m, dust, idx, tgt, sig, samples=8)
                for K in args.draws:
                    dirs[f'dust_K{K}_{tag}'], per_group = dust.estimate(idx, tgt, K, args.sigma, scaled)
                    out[f'draws_actual_K{K}'] = per_group * len(GROUPS) * len(m.h)
            curves = {name: line_search(m, dv, rows, etas) for name, dv in dirs.items()}
            best = {name: int(c[:, :32].mean(1).argmin()) for name, c in curves.items()}
            run = dict(ckpt=ckpt, batch=bi, base_loss=base[32:].mean().item(), directions={})
            for name, c in curves.items():
                tag = 'scaled' if name.endswith('scaled') else 'fixed'
                run['directions'][name] = dict(
                    best_eta=etas[best[name]], eta_at_edge=best[name] in (0, len(etas) - 1),
                    delta_vs_start=boot_ci(c[best[name], 32:] - base[32:]),
                    delta_vs_backprop=boot_ci(c[best[name], 32:] - curves['backprop'][best['backprop'], 32:]),
                    cos_backprop=cosine(dirs[name], dirs['backprop']),
                    cos_smoothed=cosine(dirs[name], dirs[f'smoothed_{tag}']),
                    cos_backprop_by_kind={k: cosine(dirs[name], dirs['backprop'], [(i, k) for i in range(12)])
                                          for k in KINDS},
                    curve=c[:, 32:].mean(1).tolist())
            out['runs'].append(run)
            out['minutes'] = (time.perf_counter() - start) / 60
            save('probe', out)
            print(json.dumps({n: (v['delta_vs_start'][0], v['cos_backprop']) for n, v in run['directions'].items()}),
                  flush=True)


CURVE_GROUPS = ((0, ('fc',)), (5, ('attn_proj', 'mlp_proj')), (11, ('fc',)))


def set_diag(m, on):
    for b in m.h:
        b.diag = on


def accumulate(acc, d, w, keys=None):
    """Running weighted sum of direction dicts, kept on the CPU."""
    d = {k: (a.cpu() * w, b.cpu() * w) for k, (a, b) in d.items() if keys is None or k in keys}
    if acc is None:
        return d
    for k, (a, b) in d.items():
        acc[k][0].add_(a)
        acc[k][1].add_(b)
    return acc


def fit_curve(ks, cos):
    """Fit the paper's cos(K) = c_max / sqrt(1 + c/K): 1/cos^2 is linear in 1/K."""
    pts = [(1 / k, 1 / c ** 2) for k, c in zip(ks, cos) if c > 0]
    if len(pts) < 3:
        return None
    x, y = torch.tensor(pts, dtype=torch.float64).T
    b = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = y.mean() - b * x.mean()
    if a <= 0 or b <= 0:
        return dict(a=a.item(), b=b.item(), note='no positive fit')
    c = (b / a).item()
    return dict(c_max=a.item() ** -0.5, c=c, draws_for_half_c_max=c / 3)


def probe2(args):
    d = tokens()
    m = load_gpt2(DATA / 'warm.pt').requires_grad_(False)
    dust = Dust(m)
    batch = d['train'][4000:4000 + 2 * args.subbatches]
    subs = [batch[j:j + 2].to(DEV).long() for j in range(0, len(batch), 2)]
    held = d['val'][:96]  # step size picked on rows 0-31, reported on rows 32-95
    scaled, w = args.sigma_mode == 'scaled', 1 / len(subs)
    start = time.perf_counter()
    out = dict(tokens=len(subs) * 2 * T, sigma=args.sigma, sigma_mode=args.sigma_mode, nrep=args.nrep,
               dust_draws=args.group_draws * len(GROUPS) * len(m.h), smooth_samples=args.smooth_samples)

    def log(msg):
        print(f'[{(time.perf_counter() - start) / 60:6.1f} min] {msg}', flush=True)

    # 1. Full-network directions, each averaged over the same 16k tokens.
    bp = same = dz = None
    for j, r in enumerate(subs):
        idx, tgt = r[:, :-1], r[:, 1:]
        bp = accumulate(bp, backprop_dir(m, dust, idx, tgt), w)
        set_diag(m, True)
        same = accumulate(same, backprop_dir(m, dust, idx, tgt), w)
        set_diag(m, False)
        dz = accumulate(dz, dust.estimate(idx, tgt, out['dust_draws'], args.sigma, scaled, args.nrep)[0], w)
        log(f'directions {j + 1}/{len(subs)}')
    gen = torch.Generator().manual_seed(0)
    rnd = {k: (torch.randn(a.shape, generator=gen), torch.randn(b.shape, generator=gen)) for k, (a, b) in bp.items()}
    dirs = dict(backprop=bp, backprop_same_token=same, dust=dz, random=rnd)
    pairs = [('dust', 'backprop'), ('dust', 'backprop_same_token'), ('backprop_same_token', 'backprop'),
             ('random', 'backprop'), ('random', 'dust')]
    out['cosine'] = {f'{a} vs {b}': cosine(dirs[a], dirs[b]) for a, b in pairs}
    out['cosine_dust_vs_same_token_by_kind'] = {k: cosine(dz, same, [(i, k) for i in range(len(m.h))]) for k in KINDS}
    save('probe2', out)
    log(f"cosines {out['cosine']}")

    # 2. Line search along each direction: in-sample (the same 16k tokens) and held-out.
    etas = torch.logspace(-2, 1, 10).tolist()
    rows = torch.cat([batch, held])
    base = seq_losses(m, rows, bs=args.eval_bs)
    curves = {name: line_search(m, dv, rows, etas, args.eval_bs) for name, dv in dirs.items()}
    n_in = len(batch)
    pick = {name: (int(c[:, :n_in].mean(1).argmin()), int(c[:, n_in:n_in + 32].mean(1).argmin()))
            for name, c in curves.items()}
    held_bp = curves['backprop'][pick['backprop'][1], n_in + 32:]
    out['line_search'] = {'etas': etas}
    for name, c in curves.items():
        bi, bo = pick[name]
        out['line_search'][name] = dict(
            in_sample=dict(best_eta=etas[bi], delta=boot_ci(c[bi, :n_in] - base[:n_in]),
                           curve=(c[:, :n_in].mean(1) - base[:n_in].mean()).tolist()),
            held_out=dict(best_eta=etas[bo], delta=boot_ci(c[bo, n_in + 32:] - base[n_in + 32:]),
                          delta_vs_backprop=boot_ci(c[bo, n_in + 32:] - held_bp),
                          curve=(c[:, n_in + 32:].mean(1) - base[n_in + 32:].mean()).tolist()))
    save('probe2', out)
    log('line search done')
    del dirs, curves, bp, same, dz, rnd

    # 3. Draw-scaling curves for single groups against exact references on the same tokens:
    #    backprop, same-token backprop, Gaussian-smoothed backprop (noise on this group only), and both.
    #    Same-token + smoothed is the exact infinite-draw limit of Dust's estimator for the group.
    ks = [k for k in (32, 64, 128, 256, 512, 1024, 2048, 4096) if k <= args.curve_max and k % args.nrep == 0]
    curve_subs = subs[:args.curve_subbatches]
    wc = 1 / len(curve_subs)
    out['curves'] = {}
    for i, group in CURVE_GROUPS:
        keys = [(i, k) for k in group]
        est, refs = {K: None for K in ks}, dict(backprop=None, same_token=None, smoothed=None, same_token_smoothed=None)
        for j, r in enumerate(curve_subs):
            idx, tgt = r[:, :-1], r[:, 1:]
            sig = dust.capture(idx, tgt, args.sigma, scaled)
            for K, g in dust.group_estimates(i, group, ks[-1], sig, args.nrep, set(ks)).items():
                est[K] = accumulate(est[K], g, wc)
            gsig = {k: sig[k] for k in keys}
            for name, diag, s in (('backprop', False, None), ('same_token', True, None),
                                  ('smoothed', False, gsig), ('same_token_smoothed', True, gsig)):
                set_diag(m, diag)
                refs[name] = accumulate(refs[name], backprop_dir(m, dust, idx, tgt, s, args.smooth_samples if s else 1),
                                        wc, keys)
                set_diag(m, False)
            log(f'curve layer {i} {"+".join(group)}: {j + 1}/{len(curve_subs)}')
        cos = {name: [cosine(est[K], ref, keys) for K in ks] for name, ref in refs.items()}
        names = list(refs)
        out['curves'][f'{i}:{"+".join(group)}'] = dict(
            K=ks, cos=cos, fit={name: fit_curve(ks, v) for name, v in cos.items()},
            reference_cos={f'{a} vs {b}': cosine(refs[a], refs[b], keys)
                           for x, a in enumerate(names) for b in names[x + 1:]})
        out['minutes'] = (time.perf_counter() - start) / 60
        save('probe2', out)
        log(f"layer {i}: cos vs same_token_smoothed {[round(c, 3) for c in cos['same_token_smoothed']]}, "
            f"vs backprop {[round(c, 3) for c in cos['backprop']]}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--stage', choices=('bench', 'warm', 'probe', 'probe2'), required=True)
    p.add_argument('--nrep', type=int, default=8, help='probe2: draws per batched forward (lower if out of memory)')
    p.add_argument('--subbatches', type=int, default=8, help='probe2: 2x1024-token sub-batches per direction')
    p.add_argument('--group-draws', type=int, default=32, help='probe2: Dust draws per jitter group (36 groups)')
    p.add_argument('--curve-max', type=int, default=1024, help='probe2: largest draw count on the scaling curves')
    p.add_argument('--curve-subbatches', type=int, default=8)
    p.add_argument('--smooth-samples', type=int, default=16, help='probe2: noise samples per smoothed reference')
    p.add_argument('--sigma-mode', choices=('scaled', 'fixed'), default='scaled')
    p.add_argument('--eval-bs', type=int, default=8)
    p.add_argument('--lr', type=float, default=6e-5)
    p.add_argument('--draws', type=int, nargs='+', default=[256, 1024])
    p.add_argument('--sigma', type=float, default=0.2)
    p.add_argument('--batches', type=int, default=3)
    p.add_argument('--max-minutes', type=float, default=60)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    dict(bench=bench, warm=warm, probe=probe, probe2=probe2)[args.stage](args)


if __name__ == '__main__':
    main()
