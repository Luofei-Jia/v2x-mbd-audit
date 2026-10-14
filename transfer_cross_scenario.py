"""Cross-scenario transfer: train on highway_2, test on urban_2 (and reverse).

Features F3+F4 (kinematics + rate + physics consistency + direction/rcv-send
coherence). RF 200 trees x 3 seeds, message-level, dedup with max(attacker).
"""
import json, glob, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = 'data/nextgen'
NUM = ['spd', 'spd_noise', 'acl', 'acl_noise', 'hed', 'hed_noise']
PROF = {'NORMAL': 0, 'CAUTIOUS': 1, 'AGGRESSIVE': 2}
ATTACKS = ['accelerationMultiplication', 'constantPositionOffset',
           'constantSpeedOffset', 'dataReplay', 'dosAttack', 'feignedBraking',
           'positionMirroring', 'randomPositionOffset', 'randomSpeedOffset',
           'reversedHeading', 'suddenConstantSpeed', 'suddenStop',
           'timeDelayAttack', 'trafficCongestionSybil', 'zeroSpeedReport']

FEATS = ['delay_ms', 's_profile', 'r_profile', 's_posx', 's_posy', 's_edge',
         'dt_s', 'rate', 'consistency', 'move', 'dt_rcv_s', 'dt_mismatch',
         'dir_consistency'] + [f'{p}_{c}' for p in ('s', 'r') for c in NUM]


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load(scenario, atk, split):
    name = f'InTAS_{scenario}_{atk}'
    fs = sorted(glob.glob(f'{ROOT}/{name}/{name}/{split}/**/*.json', recursive=True))
    rows = {}
    for f in fs:
        for m in json.load(open(f)):
            k = (m['sender_id'], m['messageID'])
            s, r = m['sender'], m['receiver']
            pos = str(s['pos']).split(',')
            d = {'sender_id': m['sender_id'], 'attacker': int(m.get('attacker', 0)),
                 'sendTime': int(m['sendTime']), 'rcvTime': int(m['rcvTime']),
                 'delay_ms': (int(m['rcvTime']) - int(m['sendTime'])) / 1e6,
                 's_profile': PROF.get(s.get('driversProfile'), -1),
                 'r_profile': PROF.get(r.get('driversProfile'), -1),
                 's_posx': fnum(pos[0]), 's_posy': fnum(pos[1]),
                 's_edge': fnum(s.get('distance_to_road_edge'))}
            for p, dct in [('s', s), ('r', r)]:
                for c in NUM:
                    d[f'{p}_{c}'] = fnum(dct.get(c))
            if k in rows:
                rows[k]['attacker'] = max(rows[k]['attacker'], d['attacker'])
            else:
                rows[k] = d
    df = pd.DataFrame(list(rows.values())).sort_values(['sender_id', 'sendTime'])
    g = df.groupby('sender_id', sort=False)
    dt_s = g['sendTime'].diff() / 1e9
    dt_rcv = g['rcvTime'].diff() / 1e9
    dx, dy = g['s_posx'].diff(), g['s_posy'].diff()
    dist = np.hypot(dx, dy)
    df['dt_s'] = dt_s
    df['dt_rcv_s'] = dt_rcv
    df['dt_mismatch'] = np.abs(dt_s - dt_rcv)
    df['rate'] = 1.0 / dt_s
    df['consistency'] = np.abs(dist / dt_s - df['s_spd'])
    df['move'] = dist
    df['dir_consistency'] = np.abs(
        (np.degrees(np.arctan2(dy, dx)) - df['s_hed'] + 180) % 360 - 180)
    return df


def fit_eval(tr, te):
    Xtr, ytr = tr[FEATS].fillna(-1).values, tr['attacker'].values
    Xte, yte = te[FEATS].fillna(-1).values, te['attacker'].values
    aucs, aprs = [], []
    for seed in range(3):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
        aprs.append(average_precision_score(yte, p))
    return float(np.mean(aucs)), float(np.mean(aprs))


def main():
    t0 = time.time()
    out = {}
    cache = {}
    for atk in ATTACKS:
        for sc in ['highway_2', 'urban_2']:
            cache[(sc, atk)] = load(sc, atk, 'Train')
        res = {}
        for src, dst in [('highway_2', 'urban_2'), ('urban_2', 'highway_2')]:
            tr = cache[(src, atk)]
            te = load(dst, atk, 'Test')
            auc, apr = fit_eval(tr, te)
            res[f'{src}_to_{dst}'] = {'auroc': round(auc, 4), 'auprc': round(apr, 4)}
        out[atk] = res
        print(f'{atk:28s} ' +
              '  '.join(f'{k}:{v["auroc"]:.3f}' for k, v in res.items()),
              flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/transfer_cross_scenario.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
