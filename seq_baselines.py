"""Sequence baselines (LSTM / Transformer encoder) on the hard attacks.

Pre-registered predictions (protocol 9b): with F3+F4 message-level features
under honest P2, sequence models should NOT rescue the hard attacks:
timeDelay stays ~0.5 (construction-invariant features), posMirroring ~0.7,
dataReplay ~0.66 (cross-sender content signal absent from these features).
Per-sender sequences (by sendTime), windows of 32, BCE per message.
"""
import json, os, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score
from sweep_hard3_f5 import load_rows, dedup, FEATS_F34, ROOT

HARD = ['dataReplay', 'positionMirroring', 'timeDelayAttack']
SEQ_LEN = 32
STRIDE = 8


def make_windows(df, seq_len=SEQ_LEN, stride=STRIDE):
    X, y = [], []
    for _, g in df.groupby('sender_id', sort=False):
        vals = g[FEATS_F34].fillna(-1).values.astype(np.float32)
        vals = np.nan_to_num(vals, nan=-1.0, posinf=0.0, neginf=0.0)
        labs = g['attacker'].values.astype(np.float32)
        if len(g) < 8:
            continue
        for i in range(0, len(g) - seq_len + 1, stride):
            X.append(vals[i:i + seq_len])
            y.append(labs[i:i + seq_len])
    return np.stack(X), np.stack(y)


class LSTMNet(nn.Module):
    def __init__(self, d_in, h=64):
        super().__init__()
        self.lstm = nn.LSTM(d_in, h, num_layers=2, batch_first=True)
        self.fc = nn.Linear(h, 1)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o).squeeze(-1)


class TFNet(nn.Module):
    def __init__(self, d_in, d=64, heads=4, layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, d))
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=256,
                                         batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.fc = nn.Linear(d, 1)

    def forward(self, x):
        h = self.proj(x) + self.pos
        return self.fc(self.enc(h)).squeeze(-1)


def train_eval(Xtr, ytr, Xte, yte, model_cls, seed, epochs=30, lr=1e-3, bs=256):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    mu = Xtr.reshape(-1, Xtr.shape[-1]).mean(0)
    sd = Xtr.reshape(-1, Xtr.shape[-1]).std(0) + 1e-6
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    model = model_cls(Xtr.shape[-1]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    n = len(Xtr)
    for ep in range(epochs):
        perm = np.random.permutation(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = torch.tensor(Xtr[idx], device=dev)
            yb = torch.tensor(ytr[idx], device=dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for i in range(0, len(Xte), bs):
            xb = torch.tensor(Xte[i:i + bs], device=dev)
            preds.append(torch.sigmoid(model(xb)).cpu().numpy())
            labs.append(yte[i:i + bs])
    p = np.concatenate(preds).ravel()
    y = np.concatenate(labs).ravel()
    return roc_auc_score(y, p), average_precision_score(y, p)


def main():
    t0 = time.time()
    out = {}
    for atk in HARD:
        name = f'InTAS_highway_2_{atk}'
        print(f'loading {atk} ...', flush=True)
        tr = dedup(load_rows(name, 'Train'))
        te = dedup(load_rows(name, 'Test'))
        tr_senders = set(tr.sender_id)
        tr = tr[~tr.sender_id.isin(set(te.sender_id))]
        te = te[~te.sender_id.isin(tr_senders)]
        Xtr, ytr = make_windows(tr)
        Xte, yte = make_windows(te, stride=SEQ_LEN)
        print(f'  train windows {len(Xtr)}, test windows {len(Xte)}', flush=True)
        res = {}
        for mname, cls in [('LSTM', LSTMNet), ('Transformer', TFNet)]:
            aucs, aprs = [], []
            for seed in range(3):
                auc, apr = train_eval(Xtr, ytr, Xte, yte, cls, seed)
                aucs.append(auc)
                aprs.append(apr)
                print(f'  {mname} seed{seed}: auc={auc:.3f} apr={apr:.3f}', flush=True)
            res[mname] = {'auroc': round(float(np.mean(aucs)), 4),
                          'auprc': round(float(np.mean(aprs)), 4),
                          'auroc_seeds': [round(float(v), 4) for v in aucs],
                          'auprc_seeds': [round(float(v), 4) for v in aprs]}
        out[atk] = res
        print(json.dumps({atk: res}, indent=1), flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/seq_baselines_seeds.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/seq_baselines_seeds.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
