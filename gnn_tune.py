"""GNN tuning run: MLP no-graph control + GraphSAGE at 200 epochs (hard3, P2).

Recovered from the 2026-10-14 session transcript; split corrected to true P2
(senders seen in the other split removed from both sides). Output:
results/gnn_tune_<attack>.json  {MLP, SAGE200}, 3 seeds averaged.
"""
import json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from stgnn_baseline import build_graph, GraphSAGE, ATTACKS
from sweep_hard3_f5 import load_rows, dedup


class MLP(nn.Module):
    def __init__(self, d_in, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, x, ei=None):
        return self.net(x).squeeze(-1)


def main():
    for atk in ATTACKS:
        t0 = time.time()
        name = f'InTAS_highway_2_{atk}'
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
        dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        xb, yb = torch.tensor(Xtr, device=dev), torch.tensor(ytr, device=dev)
        eib = torch.tensor(ei_tr, device=dev)
        xte_t = torch.tensor(Xte, device=dev)
        eite_t = torch.tensor(ei_te, device=dev)
        res = {}
        for mname, cls in [('MLP', MLP), ('SAGE200', GraphSAGE)]:
            aucs = []
            for seed in range(3):
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = cls(Xtr.shape[-1]).to(dev)
                opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                                       weight_decay=1e-4)
                lossf = nn.BCEWithLogitsLoss()
                for ep in range(200):
                    opt.zero_grad()
                    loss = lossf(model(xb, eib) if mname == 'SAGE200' else model(xb), yb)
                    loss.backward()
                    opt.step()
                model.eval()
                with torch.no_grad():
                    p = torch.sigmoid(model(xte_t, eite_t) if mname == 'SAGE200'
                                      else model(xte_t)).cpu().numpy()
                aucs.append(roc_auc_score(yte, p))
            res[mname] = round(float(np.mean(aucs)), 4)
        print(atk, res, f'({time.time()-t0:.0f}s)', flush=True)
        with open(f'results/gnn_tune_{atk}.json', 'w') as f:
            json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
