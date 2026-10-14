"""Event-level metrics + alarm burden (protocol section 6, v2 methodology).

P2 sender-disjoint, highway_2. Per attack: RF 200x3 on F34 (dataReplay:
F34+F5legit), predict Test, average probs. Events = maximal runs of
attacker=1 messages per sender with gap <= 2s. Reports message AUROC,
event AUROC (max score per event), event TPR at message FPR operating
points {10%, 1%, 0.1%}, and false alarms per vehicle-hour.
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sweep_hard3_f5 import load_rows, dedup, FEATS_F34, F5LEGIT, ROOT

ATTACKS = ['accelerationMultiplication', 'constantPositionOffset',
           'constantSpeedOffset', 'dataReplay', 'dosAttack', 'feignedBraking',
           'positionMirroring', 'randomPositionOffset', 'randomSpeedOffset',
           'reversedHeading', 'suddenConstantSpeed', 'suddenStop',
           'timeDelayAttack', 'trafficCongestionSybil', 'zeroSpeedReport']
TEST_DUR_S = 7200.0
FPR_POINTS = [0.1, 0.01, 0.001]


def get_events(df):
    """Runs of attacker=1 per sender with gap tolerance 2s.

    df must be sorted by (sender_id, sendTime) already; returns positional
    (start, end) slices into df's row order.
    """
    evs = []
    pos = 0
    for _, g in df.groupby('sender_id', sort=False):
        lab = g['attacker'].values
        t = g['sendTime'].values
        i = 0
        while i < len(lab):
            if lab[i] == 1:
                j = i
                while j + 1 < len(lab) and lab[j + 1] == 1 and \
                        (t[j + 1] - t[j]) <= 2e9:
                    j += 1
                evs.append((pos + i, pos + j))
                i = j + 1
            else:
                i += 1
        pos += len(g)
    return evs


def main():
    t0 = time.time()
    out = {}
    for atk in ATTACKS:
        name = f'InTAS_highway_2_{atk}'
        feats = FEATS_F34 + (F5LEGIT if atk == 'dataReplay' else [])
        tr = dedup(load_rows(name, 'Train'))
        te = dedup(load_rows(name, 'Test'))
        tr_senders = set(tr.sender_id)
        tr = tr[~tr.sender_id.isin(set(te.sender_id))]
        te = te[~te.sender_id.isin(tr_senders)]
        Xtr, ytr = tr[feats].fillna(-1).values, tr['attacker'].values
        Xte, yte = te[feats].fillna(-1).values, te['attacker'].values
        probs = np.zeros(len(te))
        for seed in range(3):
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                         random_state=seed)
            clf.fit(Xtr, ytr)
            probs += clf.predict_proba(Xte)[:, 1] / 3.0
        te = te.copy()
        te['score'] = probs
        evs = get_events(te)
        n_ev = len(evs)
        # message AUROC
        msg_auc = roc_auc_score(yte, probs)
        # event AUROC: max score per event vs 1; non-event messages vs 0
        ev_score, ev_lab = [], []
        in_ev = np.zeros(len(te), dtype=bool)
        ev_max = np.zeros(len(te))
        ev_id = np.full(len(te), -1, dtype=int)
        for eid, (a, b) in enumerate(evs):
            m = probs[a:b + 1].max()
            for k in range(a, b + 1):
                ev_max[k] = m
                ev_id[k] = eid
                in_ev[k] = True
        y_ev = np.where(in_ev, 1.0, 0.0)
        ev_auc = roc_auc_score(y_ev, ev_max) if in_ev.sum() > 0 else float('nan')
        # operating points on normal messages
        norm = probs[~in_ev]
        res = {'n_test': int(len(te)), 'n_events': n_ev,
               'msg_auroc': round(float(msg_auc), 4),
               'event_auroc': round(float(ev_auc), 4)}
        for fp in FPR_POINTS:
            thr = np.quantile(norm, 1.0 - fp) if len(norm) else np.inf
            # event TPR: event flagged if any message score >= thr
            tp = 0
            for eid in range(n_ev):
                if ev_max[ev_id == eid].max() >= thr:
                    tp += 1
            tpr = tp / n_ev if n_ev else float('nan')
            # false alarms per vehicle-hour: fp * normal_msgs / (n_senders * hours)
            n_snd = te.sender_id.nunique()
            fph = fp * len(norm) / (n_snd * TEST_DUR_S / 3600.0)
            res[f'fpr_{fp}'] = {'thr': round(float(thr), 4),
                                'event_tpr': round(float(tpr), 4),
                                'false_alarms_per_veh_hour': round(float(fph), 3)}
        out[atk] = res
        print(f'{atk:28s} msg={res["msg_auroc"]:.3f} ev={res["event_auroc"]:.3f} '
              f'events={n_ev}  ' +
              '  '.join(f'{fp}:{res[f"fpr_{fp}"]["event_tpr"]:.2f}/{res[f"fpr_{fp}"]["false_alarms_per_veh_hour"]:.1f}'
                        for fp in FPR_POINTS), flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/event_metrics.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/event_metrics.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
