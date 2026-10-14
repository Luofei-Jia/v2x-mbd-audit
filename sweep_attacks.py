"""Full sweep: 15 attacks x 3 protocols x 2 feature sets, message-level RF.

F1 = instantaneous kinematics (no identity, no rate/consistency)
F3 = F1 + per-sender rate + physics consistency
P1 = official temporal split; P2 = sender-disjoint; P3 = random 70/30.
Dedup by (sender_id, messageID) keeping max(attacker) on duplicates.
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

F1 = ['delay_ms', 's_profile', 'r_profile', 's_posx', 's_posy', 's_edge'] + \
     [f'{p}_{c}' for p in ('s', 'r') for c in NUM]
F3 = F1 + ['dt_s', 'rate', 'consistency', 'move']


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_split(name, split):
    fs = sorted(glob.glob(f'{ROOT}/{name}/{name}/{split}/**/*.json', recursive=True))
    rows = {}
    for f in fs:
        for m in json.load(open(f)):
            k = (m['sender_id'], m['messageID'])
            s, r = m['sender'], m['receiver']
            pos = str(s['pos']).split(',')
            d = {'sender_id': m['sender_id'], 'attacker': int(m.get('attacker', 0)),
                 'sendTime': int(m['sendTime']),
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
    dist = np.hypot(g['s_posx'].diff(), g['s_posy'].diff())
    df['dt_s'] = dt_s
    df['rate'] = 1.0 / dt_s
    df['consistency'] = np.abs(dist / dt_s - df['s_spd'])
    df['move'] = dist
    return df


def fit_eval(tr, te, feats):
    Xtr, ytr = tr[feats].fillna(-1).values, tr['attacker'].values
    Xte, yte = te[feats].fillna(-1).values, te['attacker'].values
    aucs, aprs = [], []
    for seed in range(3):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
        aprs.append(average_precision_score(yte, p))
    return float(np.mean(aucs)), float(np.mean(aprs)), aucs, aprs


def main():
    t0 = time.time()
    out = {}
    for atk in ATTACKS:
        name = f'InTAS_highway_2_{atk}'
        tr, te = load_split(name, 'Train'), load_split(name, 'Test')
        tr2 = tr[~tr.sender_id.isin(set(te.sender_id))]
        te2 = te[~te.sender_id.isin(set(tr.sender_id))]
        merged = pd.concat([tr, te])
        idx = np.random.RandomState(42).permutation(len(merged))
        cut = int(0.7 * len(merged))
        tr3, te3 = merged.iloc[idx[:cut]], merged.iloc[idx[cut:]]
        res = {'att_rate': round(float(te.attacker.mean()), 3)}
        for pname, (a, b) in [('P1', (tr, te)), ('P2', (tr2, te2)), ('P3', (tr3, te3))]:
            for fname, feats in [('F1', F1), ('F3', F3)]:
                if fname == 'F1' and pname == 'P2':
                    continue
                auc, apr, aucs, aprs = fit_eval(a, b, feats)
                res[f'{pname}_{fname}'] = {
                    'auroc': round(auc, 4), 'auprc': round(apr, 4),
                    'auroc_seeds': [round(v, 4) for v in aucs],
                    'auprc_seeds': [round(v, 4) for v in aprs]}
        out[atk] = res
        print(f'{atk:28s} att={res["att_rate"]:.2f}  ' +
              '  '.join(f'{k}:{v["auroc"]:.3f}' for k, v in res.items()
                        if isinstance(v, dict)),
              flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/sweep_attacks_seeds.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/sweep_attacks_seeds.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
