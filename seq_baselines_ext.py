"""LSTM/Transformer coverage over the remaining 12 attacks (protocol 9n).

Same pipeline and hyper-parameters as seq_baselines.py (P2, F34 features,
32-message per-sender windows, 3 seeds); defensive completeness for the
Table IV scope question. Output: results/seq_baselines_ext_seeds.json
"""
import json
import os
import time

import numpy as np

from sweep_hard3_f5 import load_rows, dedup
from seq_baselines import LSTMNet, TFNet, train_eval, make_windows, SEQ_LEN

EXT = ['dosAttack', 'zeroSpeedReport', 'constantSpeedOffset',
       'randomSpeedOffset', 'randomPositionOffset', 'trafficCongestionSybil',
       'suddenConstantSpeed', 'reversedHeading', 'feignedBraking',
       'suddenStop', 'accelerationMultiplication', 'constantPositionOffset']


def main():
    t0 = time.time()
    out = {}
    for atk in EXT:
        ta = time.time()
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
                print(f'  {mname} seed{seed}: auc={auc:.3f} apr={apr:.3f}',
                      flush=True)
            res[mname] = {'auroc': round(float(np.mean(aucs)), 4),
                          'auprc': round(float(np.mean(aprs)), 4),
                          'auroc_seeds': [round(float(v), 4) for v in aucs],
                          'auprc_seeds': [round(float(v), 4) for v in aprs]}
        out[atk] = res
        print(json.dumps({atk: res}, indent=1), flush=True)
        print(f'[{atk}] done in {time.time() - ta:.0f}s '
              f'(total {time.time() - t0:.0f}s)', flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/seq_baselines_ext_seeds.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/seq_baselines_ext_seeds.json '
          f'({time.time() - t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
