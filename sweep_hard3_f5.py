"""Hard-3 F5 sweep: receiver-view features under P1/P2, RF 200x3 seeds.

F5legit = dist_sr, close_rate (per receiver-sender pair), lat_off (vs r_hed),
          collision (exact content match, different sender, <=6s)
F5art   = rcv_lag (timeDelay only, post-processing artifact, reported apart)
Baseline F3+F4 exactly as sweep_f4/transfer_cross_scenario.
Per-receiver features computed pre-dedup, aggregated max over
(sender_id, messageID) copies; F3/F4 diffs on deduped stream as before.
"""
import json, glob, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = 'data/nextgen'
NUM = ['spd', 'spd_noise', 'acl', 'acl_noise', 'hed', 'hed_noise']
PROF = {'NORMAL': 0, 'CAUTIOUS': 1, 'AGGRESSIVE': 2}
HARD3 = ['dataReplay', 'positionMirroring', 'timeDelayAttack']

FEATS_F34 = ['delay_ms', 's_profile', 'r_profile', 's_posx', 's_posy', 's_edge',
             'dt_s', 'rate', 'consistency', 'move', 'dt_rcv_s', 'dt_mismatch',
             'dir_consistency'] + [f'{p}_{c}' for p in ('s', 'r') for c in NUM]
F5LEGIT = ['dist_sr', 'close_rate', 'lat_off', 'collision']
F5ART = ['rcv_lag']


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_rows(name, split):
    """Undeduped rows with per-receiver F5 computed in place."""
    fs = sorted(glob.glob(f'{ROOT}/{name}/{name}/{split}/**/*.json', recursive=True))
    rows = []
    for f in fs:
        rcv = f.split('/')[-1][4:-5]
        for m in json.load(open(f)):
            s, r = m['sender'], m['receiver']
            ps = str(s['pos']).split(',')
            pr = str(r['pos']).split(',')
            rows.append({
                'receiver': rcv, 'sender_id': m['sender_id'],
                'messageID': m['messageID'],
                'attacker': int(m.get('attacker', 0)),
                'sendTime': int(m['sendTime']), 'rcvTime': int(m['rcvTime']),
                'delay_ms': (int(m['rcvTime']) - int(m['sendTime'])) / 1e6,
                's_profile': PROF.get(s.get('driversProfile'), -1),
                'r_profile': PROF.get(r.get('driversProfile'), -1),
                's_posx': fnum(ps[0]), 's_posy': fnum(ps[1]),
                's_edge': fnum(s.get('distance_to_road_edge')),
                's_pos_str': str(s['pos']),
                's_spd': fnum(s.get('spd')), 's_acl': fnum(s.get('acl')),
                's_hed': fnum(s.get('hed')),
                'r_posx': fnum(pr[0]), 'r_posy': fnum(pr[1]),
                'r_hed': fnum(r.get('hed')),
            })
            for p, dct in [('s', s), ('r', r)]:
                for c in NUM:
                    rows[-1][f'{p}_{c}'] = fnum(dct.get(c))
    for r in rows:
        r['dist_sr'] = np.hypot(r['s_posx'] - r['r_posx'], r['s_posy'] - r['r_posy'])
        rh = np.radians(r['r_hed'])
        dx, dy = r['s_posx'] - r['r_posx'], r['s_posy'] - r['r_posy']
        r['lat_off'] = abs(-dx * np.sin(rh) + dy * np.cos(rh))
        r['close_rate'] = np.nan
        r['collision'] = 0.0
        r['rcv_lag'] = np.nan
    by_rcv = {}
    for r in rows:
        by_rcv.setdefault(r['receiver'], []).append(r)
    for rcv, rs in by_rcv.items():
        rs.sort(key=lambda x: x['rcvTime'])
        recent = {}
        prev = {}
        for i, x in enumerate(rs):
            key = (x['s_pos_str'], x['s_spd'], x['s_acl'], x['s_hed'])
            t = x['rcvTime']
            if key in recent and recent[key][1] != x['sender_id'] and recent[key][0] >= t - 6e9:
                x['collision'] = 1.0
            for k in list(recent.keys()):
                if recent[k][0] < t - 6e9:
                    del recent[k]
            if key not in recent:
                recent[key] = (t, x['sender_id'])
            elif recent[key][1] == x['sender_id']:
                recent[key] = (t, x['sender_id'])
            p = prev.get(x['sender_id'])
            if p is not None:
                dt = (t - p['rcvTime']) / 1e9
                if 0 < dt < 10:
                    dd = abs(x['dist_sr'] - p['dist_sr'])
                    x['close_rate'] = dd / dt
            prev[x['sender_id']] = x
            j = i - 1
            while j >= 0 and rs[j]['sender_id'] == x['sender_id']:
                j -= 1
            k = i + 1
            while k < len(rs) and rs[k]['sender_id'] == x['sender_id']:
                k += 1
            if j >= 0 and k < len(rs):
                t0, t1 = rs[j]['rcvTime'] / 1e9, rs[k]['rcvTime'] / 1e9
                tm = t / 1e9
                if t1 > t0:
                    w = (tm - t0) / (t1 - t0)
                    ex = rs[j]['r_posx'] + (rs[k]['r_posx'] - rs[j]['r_posx']) * w
                    ey = rs[j]['r_posy'] + (rs[k]['r_posy'] - rs[j]['r_posy']) * w
                    x['rcv_lag'] = np.hypot(x['r_posx'] - ex, x['r_posy'] - ey)
    return rows


def dedup(rows):
    """First-copy base + max-aggregated F5; then per-sender F3/F4 diffs."""
    base = {}
    agg = {c: {} for c in F5LEGIT + F5ART}
    order = []
    for r in rows:
        k = (r['sender_id'], r['messageID'])
        if k not in base:
            base[k] = r
            order.append(k)
        else:
            base[k]['attacker'] = max(base[k]['attacker'], r['attacker'])
        for c in F5LEGIT + F5ART:
            v = r[c]
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                agg[c][k] = max(agg[c].get(k, -np.inf), v)
    cols = ['sender_id', 'attacker', 'sendTime', 'rcvTime', 'delay_ms',
            's_profile', 'r_profile', 's_posx', 's_posy', 's_edge',
            's_spd', 's_hed'] + [f'{p}_{c}' for p in ('s', 'r') for c in NUM]
    df = pd.DataFrame([{c: base[k][c] for c in cols} for k in order])
    for c in F5LEGIT + F5ART:
        df[c] = [agg[c].get(k, np.nan) for k in order]
    df = df.sort_values(['sender_id', 'sendTime'])
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
    for atk in HARD3:
        name = f'InTAS_highway_2_{atk}'
        print(f'loading {atk} ...', flush=True)
        tr = dedup(load_rows(name, 'Train'))
        te = dedup(load_rows(name, 'Test'))
        tr2 = tr[~tr.sender_id.isin(set(te.sender_id))]
        te2 = te[~te.sender_id.isin(set(tr.sender_id))]
        res = {'att_rate': round(float(te.attacker.mean()), 3)}
        featsets = [('F34', FEATS_F34),
                    ('F34+F5legit', FEATS_F34 + F5LEGIT)]
        if atk == 'timeDelayAttack':
            featsets.append(('F34+F5legit+rcv_lag(artifact)',
                             FEATS_F34 + F5LEGIT + F5ART))
        for pname, (a, b) in [('P1', (tr, te)), ('P2', (tr2, te2))]:
            for fname, feats in featsets:
                auc, apr, aucs, aprs = fit_eval(a, b, feats)
                res[f'{pname}_{fname}'] = {
                    'auroc': round(auc, 4), 'auprc': round(apr, 4),
                    'auroc_seeds': [round(v, 4) for v in aucs],
                    'auprc_seeds': [round(v, 4) for v in aprs]}
        out[atk] = res
        print(f'{atk:20s} ' + '  '.join(f'{k}:{v["auroc"]:.3f}' for k, v in res.items()
                                        if isinstance(v, dict)), flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/sweep_hard3_f5_seeds.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/sweep_hard3_f5_seeds.json ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
