"""ST-GNN baseline (protocol 9h): self-built adjacency, 2-layer GraphSAGE.

Nodes = deduped messages. Edges: spatial (dt<=2s, dist<300m) + temporal
(consecutive same-sender). Node features = F34 (standardized). P2, highway_2,
hard-2 + dataReplay, 3 seeds. Full-batch BCE, mean aggregation.
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sweep_hard3_f5 import load_rows, dedup, FEATS_F34, ROOT

ATTACKS = ['dataReplay', 'positionMirroring', 'timeDelayAttack']
HID = 64
SPAT_DT = 2e9
SPAT_R = 300.0


def build_graph(df):
    """Return (X, edge_index) on positional node order."""
    X = df[FEATS_F34].fillna(-1).values.astype(np.float32)
    X = np.nan_to_num(X, nan=-1.0, posinf=0.0, neginf=0.0)
    n = len(df)
    t = df['sendTime'].values
    px = df['s_posx'].values
    py = df['s_posy'].values
    edges = set()
    # temporal edges: consecutive same sender (df sorted by sender_id, sendTime)
    sid = df['sender_id'].values
    for i in range(1, n):
        if sid[i] == sid[i - 1]:
            edges.add((i - 1, i))
            edges.add((i, i - 1))
    # spatial edges via sliding window over sendTime order
    order = np.argsort(t)
    active = []
    b = 0
    for i in order:
        while b < len(active) and t[active[b]] < t[i] - SPAT_DT:
            b += 1
        for j in active[b:]:
            d2 = (px[i] - px[j]) ** 2 + (py[i] - py[j]) ** 2
            if d2 < SPAT_R ** 2:
                edges.add((i, j))
                edges.add((j, i))
        active.append(i)
    ei = np.array(sorted(edges), dtype=np.int64).T
    return X, ei


class GraphSAGE(nn.Module):
    def __init__(self, d_in, h=HID, layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(nn.Linear((d_in if i == 0 else h) * 2, h))
        self.fc = nn.Linear(h, 1)

    def forward(self, x, ei):
        for lin in self.layers:
            src, dst = ei[0], ei[1]
            agg = torch.zeros_like(x)
            agg = agg.index_add(0, dst, x[src])
            cnt = torch.zeros(x.shape[0], 1, device=x.device)
            cnt = cnt.index_add(0, dst, torch.ones(len(src), 1, device=x.device))
            agg = agg / cnt.clamp(min=1)
            x = torch.relu(lin(torch.cat([x, agg], dim=-1)))
        return self.fc(x).squeeze(-1)


def main():
    t0 = time.time()
    out = {}
    for atk in ATTACKS:
        name = f'InTAS_highway_2_{atk}'
        print(f'loading {atk} ...', flush=True)
        tr = dedup(load_rows(name, 'Train'))
        te = dedup(load_rows(name, 'Test'))
        tr_senders = set(tr.sender_id)
        tr = tr[~tr.sender_id.isin(set(te.sender_id))]
        te = te[~te.sender_id.isin(tr_senders)]
        Xtr, ei_tr = build_graph(tr)
        Xte, ei_te = build_graph(te)
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
        ytr = tr['attacker'].values.astype(np.float32)
        yte = te['attacker'].values.astype(np.float32)
        print(f'  nodes {len(Xtr)}/{len(Xte)} edges {ei_tr.shape[1]}/{ei_te.shape[1]}',
              flush=True)
        aucs, aprs = [], []
        for seed in range(3):
            torch.manual_seed(seed)
            np.random.seed(seed)
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
            xb = torch.tensor(Xtr, device=dev)
            yb = torch.tensor(ytr, device=dev)
            eib = torch.tensor(ei_tr, device=dev)
            model = GraphSAGE(Xtr.shape[-1]).to(dev)
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            lossf = nn.BCEWithLogitsLoss()
            for ep in range(60):
                opt.zero_grad()
                loss = lossf(model(xb, eib), yb)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                p = torch.sigmoid(model(torch.tensor(Xte, device=dev),
                                        torch.tensor(ei_te, device=dev))).cpu().numpy()
            auc = roc_auc_score(yte, p)
            apr = average_precision_score(yte, p)
            aucs.append(auc)
            aprs.append(apr)
            print(f'  seed{seed}: auc={auc:.3f} apr={apr:.3f}', flush=True)
        out[atk] = {'auroc': round(float(np.mean(aucs)), 4),
                    'auprc': round(float(np.mean(aprs)), 4),
                    'auroc_seeds': [round(float(v), 4) for v in aucs],
                    'auprc_seeds': [round(float(v), 4) for v in aprs]}
        print(json.dumps({atk: out[atk]}, indent=1), flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/stgnn_baseline_seeds.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/stgnn_baseline_seeds.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
