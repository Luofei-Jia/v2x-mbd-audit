"""Hard-3 source-code-informed signature audit (protocol 9c).

For timeDelayAttack / dataReplay / positionMirroring (highway_2 Train),
compute per-receiver signature features BEFORE dedup, aggregate with
max over (sender_id, messageID), and report single-feature AUROC.

Signatures (registered in protocol 9c):
  timeDelay:  rcv_lag  = receiver_pos residual vs local median of other
                         senders' receiver_pos (+-3s window)  [ARTIFACT]
              dist_sr  = ||sender_pos - receiver_pos||  (inflation check:
                         expect NO separation -> real-world signal erased)
              delay_ms = rcvTime - sendTime  (expect NO separation)
  dataReplay: collision = exact content match (pos str, spd, acl, hed)
                         with earlier msg (<=6s) from different sender_id
              consis   = per-(receiver,sender) |dist/dt - s_spd| (F3 ref)
  posMirror:  close_rate = |d(dist_sr)/dt| per (receiver,sender) pair
              dist_sr  (weak reference)
"""
import json, glob, os, time
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = 'data/nextgen'
SC = 'highway_2'


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_attack(atk):
    name = f'InTAS_{SC}_{atk}'
    fs = sorted(glob.glob(f'{ROOT}/{name}/{name}/Train/*/veh_*.json'))
    recs = []
    for f in fs:
        rcv = f.split('/')[-1][4:-5]
        for m in json.load(open(f)):
            s, r = m['sender'], m['receiver']
            pos_s = str(s['pos']).split(',')
            pos_r = str(r['pos']).split(',')
            recs.append({
                'receiver': rcv,
                'sender_id': m['sender_id'],
                'messageID': m['messageID'],
                'attacker': int(m.get('attacker', 0)),
                'sendTime': int(m['sendTime']),
                'rcvTime': int(m['rcvTime']),
                's_posx': fnum(pos_s[0]), 's_posy': fnum(pos_s[1]),
                'r_posx': fnum(pos_r[0]), 'r_posy': fnum(pos_r[1]),
                's_pos_str': str(s['pos']),
                's_spd': fnum(s.get('spd')), 's_acl': fnum(s.get('acl')),
                's_hed': fnum(s.get('hed')),
                'r_hed': fnum(r.get('hed')),
            })
    return recs


def dedup_auc(rows, score_fn, agg='max'):
    """score_fn(row) -> float or NaN; aggregate per (sender_id,messageID)."""
    best, lab = {}, {}
    for r in rows:
        k = (r['sender_id'], r['messageID'])
        v = score_fn(r)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k not in best or v > best[k]:
            best[k] = v
        lab[k] = max(lab.get(k, 0), r['attacker'])
    scores = np.array([best[k] for k in lab])
    labels = np.array([lab[k] for k in lab])
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float('nan'), len(labels)
    return roc_auc_score(labels, scores), len(labels)


def timeDelay_diag(rows):
    print(f'timeDelayAttack: {len(rows)} rows')
    by_rcv = {}
    for r in rows:
        by_rcv.setdefault(r['receiver'], []).append(r)
    # per-message scores
    for r in rows:
        r['delay_ms'] = (r['rcvTime'] - r['sendTime']) / 1e6
        r['dist_sr'] = np.hypot(r['s_posx'] - r['r_posx'], r['s_posy'] - r['r_posy'])
        r['rcv_lag'] = np.nan
    for rcv, rs in by_rcv.items():
        rs.sort(key=lambda x: x['rcvTime'])
        for i, x in enumerate(rs):
            # bracket with adjacent messages (any sender) in rcvTime order
            j = i - 1
            while j >= 0 and rs[j]['sender_id'] == x['sender_id']:
                j -= 1
            k = i + 1
            while k < len(rs) and rs[k]['sender_id'] == x['sender_id']:
                k += 1
            if j >= 0 and k < len(rs):
                t0, t1 = rs[j]['rcvTime'] / 1e9, rs[k]['rcvTime'] / 1e9
                tm = x['rcvTime'] / 1e9
                if t1 > t0:
                    w = (tm - t0) / (t1 - t0)
                    ex = rs[j]['r_posx'] + (rs[k]['r_posx'] - rs[j]['r_posx']) * w
                    ey = rs[j]['r_posy'] + (rs[k]['r_posy'] - rs[j]['r_posy']) * w
                    x['rcv_lag'] = np.hypot(x['r_posx'] - ex, x['r_posy'] - ey)
    out = {}
    out['delay_ms_auc'] = dedup_auc(rows, lambda r: r['delay_ms'])[0]
    out['dist_sr_auc'] = dedup_auc(rows, lambda r: r['dist_sr'])[0]
    out['rcv_lag_auc'] = dedup_auc(rows, lambda r: r['rcv_lag'])[0]
    # marginal check: attacker vs normal delay_ms / dist_sr
    a = [r['delay_ms'] for r in rows if r['attacker'] == 1]
    n = [r['delay_ms'] for r in rows if r['attacker'] == 0]
    out['delay_ms_mean_att_vs_norm'] = [float(np.mean(a)), float(np.mean(n))]
    a = [r['dist_sr'] for r in rows if r['attacker'] == 1]
    n = [r['dist_sr'] for r in rows if r['attacker'] == 0]
    out['dist_sr_mean_att_vs_norm'] = [float(np.mean(a)), float(np.mean(n))]
    a = [r['rcv_lag'] for r in rows if r['attacker'] == 1]
    n = [r['rcv_lag'] for r in rows if r['attacker'] == 0]
    out['rcv_lag_mean_att_vs_norm'] = [float(np.nanmean(a)), float(np.nanmean(n))]
    out['rcv_lag_p90_att_vs_norm'] = [float(np.nanpercentile(a, 90)), float(np.nanpercentile(n, 90))]
    out['rcv_lag_coverage'] = float(np.mean([~np.isnan(r['rcv_lag']) for r in rows]))
    return out


def dataReplay_diag(rows):
    print(f'dataReplay: {len(rows)} rows')
    by_rcv = {}
    for r in rows:
        by_rcv.setdefault(r['receiver'], []).append(r)
    for r in rows:
        r['collision'] = 0.0
        r['collision_near'] = 0.0
        r['consis'] = np.nan
    for rcv, rs in by_rcv.items():
        rs.sort(key=lambda x: x['rcvTime'])
        # key -> (last_time, sender_id); collision only across different senders
        recent = {}
        prev = {}
        for x in rs:
            key = (x['s_pos_str'], x['s_spd'], x['s_acl'], x['s_hed'])
            key_near = (round(x['s_posx'], 3), round(x['s_posy'], 3),
                        round(x['s_spd'], 3), round(x['s_acl'], 3),
                        round(x['s_hed'], 3))
            t = x['rcvTime']
            if key in recent and recent[key][1] != x['sender_id'] and recent[key][0] >= t - 6e9:
                x['collision'] = 1.0
            if key_near in recent and recent[key_near][1] != x['sender_id'] and recent[key_near][0] >= t - 6e9:
                x['collision_near'] = 1.0
            for k in list(recent.keys()):
                if recent[k][0] < t - 6e9:
                    del recent[k]
            if key not in recent:
                recent[key] = (t, x['sender_id'])
            elif recent[key][1] == x['sender_id']:
                recent[key] = (t, x['sender_id'])
            if key_near not in recent:
                recent[key_near] = (t, x['sender_id'])
            elif recent[key_near][1] == x['sender_id']:
                recent[key_near] = (t, x['sender_id'])
            # per-(receiver,sender) F3-style consistency
            p = prev.get(x['sender_id'])
            if p is not None:
                dt = (t - p['rcvTime']) / 1e9
                if 0 < dt < 10:
                    dist = np.hypot(x['s_posx'] - p['s_posx'], x['s_posy'] - p['s_posy'])
                    x['consis'] = abs(dist / dt - x['s_spd'])
            prev[x['sender_id']] = x
    out = {}
    out['collision_auc'] = dedup_auc(rows, lambda r: r['collision'])[0]
    out['collision_near_auc'] = dedup_auc(rows, lambda r: r['collision_near'])[0]
    out['consis_auc'] = dedup_auc(rows, lambda r: r['consis'])[0]
    # collision recall on attacker messages (deduped)
    best, bestn, lab = {}, {}, {}
    for r in rows:
        k = (r['sender_id'], r['messageID'])
        best[k] = max(best.get(k, 0.0), r['collision'])
        bestn[k] = max(bestn.get(k, 0.0), r['collision_near'])
        lab[k] = max(lab.get(k, 0), r['attacker'])
    att = [best[k] for k in lab if lab[k] == 1]
    out['collision_recall_on_attacker'] = float(np.mean(att))
    attn = [bestn[k] for k in lab if lab[k] == 1]
    out['collision_near_recall_on_attacker'] = float(np.mean(attn))
    nrm = [bestn[k] for k in lab if lab[k] == 0]
    out['collision_near_fpr_on_normal'] = float(np.mean(nrm))
    out['n_attacker_msgs'] = int(sum(lab.values()))
    return out


def posMirror_diag(rows):
    print(f'positionMirroring: {len(rows)} rows')
    for r in rows:
        r['dist_sr'] = np.hypot(r['s_posx'] - r['r_posx'], r['s_posy'] - r['r_posy'])
        r['close_rate'] = np.nan
        r['lat_off'] = np.nan
    prev = {}
    for r in sorted(rows, key=lambda x: x['rcvTime']):
        k = (r['receiver'], r['sender_id'])
        p = prev.get(k)
        if p is not None:
            dt = (r['rcvTime'] - p['rcvTime']) / 1e9
            if 0 < dt < 10:
                dd = abs(r['dist_sr'] - p['dist_sr'])
                r['close_rate'] = dd / dt
        prev[k] = r
    for r in rows:
        # lateral offset of claimed sender pos relative to receiver heading
        rh = np.radians(r['r_hed'])
        dx, dy = r['s_posx'] - r['r_posx'], r['s_posy'] - r['r_posy']
        r['lat_off'] = abs(-dx * np.sin(rh) + dy * np.cos(rh))
    out = {}
    out['close_rate_auc'] = dedup_auc(rows, lambda r: r['close_rate'])[0]
    out['dist_sr_auc'] = dedup_auc(rows, lambda r: r['dist_sr'])[0]
    out['lat_off_auc'] = dedup_auc(rows, lambda r: r['lat_off'])[0]
    a = [r['close_rate'] for r in rows if r['attacker'] == 1 and not np.isnan(r['close_rate'])]
    n = [r['close_rate'] for r in rows if r['attacker'] == 0 and not np.isnan(r['close_rate'])]
    out['close_rate_mean_att_vs_norm'] = [float(np.mean(a)), float(np.mean(n))]
    a = [r['lat_off'] for r in rows if r['attacker'] == 1]
    n = [r['lat_off'] for r in rows if r['attacker'] == 0]
    out['lat_off_mean_att_vs_norm'] = [float(np.mean(a)), float(np.mean(n))]
    out['lat_off_p90_att_vs_norm'] = [float(np.percentile(a, 90)), float(np.percentile(n, 90))]
    return out


def main():
    t0 = time.time()
    out = {}
    rows = load_attack('timeDelayAttack')
    out['timeDelayAttack'] = timeDelay_diag(rows)
    print(json.dumps(out['timeDelayAttack'], indent=1), flush=True)
    del rows
    rows = load_attack('dataReplay')
    out['dataReplay'] = dataReplay_diag(rows)
    print(json.dumps(out['dataReplay'], indent=1), flush=True)
    del rows
    rows = load_attack('positionMirroring')
    out['positionMirroring'] = posMirror_diag(rows)
    print(json.dumps(out['positionMirroring'], indent=1), flush=True)
    os.makedirs('results', exist_ok=True)
    with open('results/hard3_signatures.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    main()
