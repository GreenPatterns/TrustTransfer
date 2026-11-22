"""Neural transfer baselines: DANN, V-REx, and an inductive relational GNN
with optional multi-platform Deep Graph Infomax pretraining. All are
identity-free: nothing is indexed by node id across graphs."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

import tt

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H = 64


def _t(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=DEV)


def _mlp(d_in, h=H):
    return nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())


def _balanced_bce(logit, y):
    pw = (y == 0).sum().clamp(min=1) / (y == 1).sum().clamp(min=1)
    return F.binary_cross_entropy_with_logits(logit, y, pos_weight=pw)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def dann(xs, ys, xt, seed, steps=1500, bs=2048):
    """Ganin et al. 2016: shared encoder, label head, adversarial domain head."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    enc, head = _mlp(xs.shape[1]).to(DEV), nn.Linear(H, 1).to(DEV)
    dom = nn.Sequential(nn.Linear(H, H), nn.ReLU(), nn.Linear(H, 1)).to(DEV)
    opt = torch.optim.Adam([*enc.parameters(), *head.parameters(), *dom.parameters()], lr=1e-3)
    xs_t, ys_t, xt_t = _t(xs), _t(ys), _t(xt)
    for step in range(steps):
        i, j = rng.integers(0, len(xs), bs), rng.integers(0, len(xt), bs)
        lam = 2 / (1 + np.exp(-10 * step / steps)) - 1
        hs, ht = enc(xs_t[i]), enc(xt_t[j])
        z = dom(_GradReverse.apply(torch.cat([hs, ht]), lam)).squeeze(1)
        loss = _balanced_bce(head(hs).squeeze(1), ys_t[i]) + F.binary_cross_entropy_with_logits(
            z, torch.cat([torch.ones(bs, device=DEV), torch.zeros(bs, device=DEV)]))
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return head(enc(xt_t)).squeeze(1).cpu().numpy()


def vrex(envs, xt, seed, steps=1500, bs=2048, beta=10.0, warmup=300):
    """Krueger et al. 2021: penalise the variance of per-source risks."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = nn.Sequential(_mlp(xt.shape[1]), nn.Linear(H, 1)).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    envs = [(_t(x), _t(y)) for x, y in envs]
    for step in range(steps):
        risks = []
        for x, y in envs:
            i = rng.integers(0, len(x), bs)
            risks.append(_balanced_bce(net(x[i]).squeeze(1), y[i]))
        risks = torch.stack(risks)
        loss = risks.mean() + (beta if step >= warmup else 0.0) * risks.var()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return net(_t(xt)).squeeze(1).cpu().numpy()


# ------------------------------------------------------------------ GNN
def graph_inputs(view):
    """Node features and six relations from what the view may observe."""
    g = view.g
    out = np.bincount(g.s[view.topo], minlength=g.n)
    inn = np.bincount(g.d[view.topo], minlength=g.n)
    po, no, pi, ni, _ = tt.node_sign_stats(g, view.signs)
    x = np.column_stack([np.log1p(out), np.log1p(inn), np.log1p(po), np.log1p(no),
                         (po + 1) / (po + no + 2), np.log1p(pi), np.log1p(ni), (pi + 1) / (pi + ni + 2)])
    sy = g.y[view.signs]
    rels = []
    for s, d in ((g.s[view.topo], g.d[view.topo]),
                 (g.s[view.signs][sy == 1], g.d[view.signs][sy == 1]),
                 (g.s[view.signs][sy == 0], g.d[view.signs][sy == 0])):
        rels += [np.stack([s, d]), np.stack([d, s])]
    ei = [torch.as_tensor(r, dtype=torch.long, device=DEV) for r in rels]
    e = view.score
    return {"x": _t(tt.zscore(x)), "ei": ei, "u": torch.as_tensor(g.s[e], device=DEV),
            "v": torch.as_tensor(g.d[e], device=DEV), "ef": _t(tt.zscore(tt.features(view))),
            "y": _t(g.y[e])}


class Encoder(nn.Module):
    def __init__(self, d_in=8, layers=2, n_rel=6):
        super().__init__()
        self.inp = nn.Linear(d_in, H)
        self.convs = nn.ModuleList(nn.ModuleList(SAGEConv(H, H, root_weight=False) for _ in range(n_rel))
                                   for _ in range(layers))
        self.roots = nn.ModuleList(nn.Linear(H, H) for _ in range(layers))

    def forward(self, x, ei):
        h = F.relu(self.inp(x))
        for convs, root in zip(self.convs, self.roots):
            h = F.relu(root(h) + sum(c(h, e) for c, e in zip(convs, ei)))
        return h


class EdgeModel(nn.Module):
    def __init__(self, d_edge=15):
        super().__init__()
        self.enc = Encoder()
        self.dec = nn.Sequential(nn.Linear(2 * H + d_edge, H), nn.ReLU(), nn.Linear(H, 1))

    def forward(self, gi, idx=None):
        h = self.enc(gi["x"], gi["ei"])
        sel = slice(None) if idx is None else idx
        z = torch.cat([h[gi["u"][sel]], h[gi["v"][sel]], gi["ef"][sel]], 1)
        return self.dec(z).squeeze(1)


def dgi_pretrain(enc, graphs, seed, steps=200, sample=20000):
    """Deep Graph Infomax (Velickovic et al. 2019) over several unlabeled graphs."""
    rng = np.random.default_rng(seed)
    w = nn.Parameter(torch.empty(H, H, device=DEV))
    nn.init.xavier_uniform_(w)
    opt = torch.optim.Adam([*enc.parameters(), w], lr=1e-3)
    for step in range(steps):
        gi = graphs[step % len(graphs)]
        x = gi["x"]
        pos = enc(x, gi["ei"])
        neg = enc(x[torch.as_tensor(rng.permutation(len(x)), device=DEV)], gi["ei"])
        summary = torch.sigmoid(pos.mean(0))
        k = torch.as_tensor(rng.integers(0, len(x), min(sample, len(x))), device=DEV)
        lp, ln = pos[k] @ w @ summary, neg[k] @ w @ summary
        loss = F.binary_cross_entropy_with_logits(lp, torch.ones_like(lp)) + \
            F.binary_cross_entropy_with_logits(ln, torch.zeros_like(ln))
        opt.zero_grad()
        loss.backward()
        opt.step()


def gnn(train_gi, test_gis, seed, pretrain=None, steps=300, bs=8192):
    """Train on one view's scored edges, score each test view's edges."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = EdgeModel().to(DEV)
    if pretrain:
        dgi_pretrain(model.enc, pretrain, seed)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    m = len(train_gi["y"])
    for _ in range(steps):
        idx = torch.as_tensor(rng.integers(0, m, min(bs, m)), device=DEV)
        loss = _balanced_bce(model(train_gi, idx), train_gi["y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        return [model(gi).cpu().numpy() for gi in test_gis]
