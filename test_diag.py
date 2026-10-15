"""Check that same-token attention returns sum_t d loss_t / d y_t (Dust's target), on a tiny CPU model."""
import torch

from run import GPT, set_diag

torch.manual_seed(0)
m = GPT(layers=2, d=48).double()
idx, tgt = torch.randint(0, 50257, (1, 8)), torch.randint(0, 50257, (1, 8))
site = m.h[0].fc
cap = {}


def keep(mod, args, out):  # returns None so the output is left unchanged
    out.retain_grad()
    cap['y'] = out


site.register_forward_hook(keep)

# Brute force: row t of d loss_t / d y, one backward per token.
brute = torch.zeros(8, 192, dtype=torch.float64)
for t in range(8):
    m.zero_grad()
    m.losses(idx, tgt, amp=False)[0, t].backward()
    brute[t] = cap['y'].grad[0, t]

set_diag(m, True)
m.zero_grad()
diag_loss = m.losses(idx, tgt, amp=False)
diag_loss.sum().backward()
set_diag(m, False)
fast = cap['y'].grad[0]

assert torch.allclose(diag_loss, m.losses(idx, tgt, amp=False)), 'same-token mode changed forward values'
assert torch.allclose(fast, brute, rtol=1e-4, atol=1e-7), (fast - brute).abs().max()  # losses cast logits to float32
print('ok: same-token gradient matches brute force; max err', (fast - brute).abs().max().item())
