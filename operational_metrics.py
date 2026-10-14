"""Operational metrics: detection latency, online stability, inference cost
(protocol 9m). Config identical to event_metrics (protocol 9i): P2, RF 200x3,
F34 (dataReplay +F5legit), highway_2; event = per-sender attack run, <=2s gap.
Latency thresholds = quantiles of non-event (normal) scores at message FPR
{1%, 0.1%}; stability = 600s rcvTime windows; inference cost = predict_proba
timing (n_jobs=1 and -1).
"""
import json, os, pickle, time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sweep_hard3_f5 import load_rows, dedup, FEATS_F34, F5LEGIT
from event_metrics import get_events

ATTACKS = ['accelerationMultiplication', 'constantPositionOffset',
           'constantSpeedOffset', 'dataReplay', 'dosAttack', 'feignedBraking',
           'positionMirroring', 'randomPositionOffset', 'randomSpeedOffset',
           'reversedHeading', 'suddenConstantSpeed', 'suddenStop',
           'timeDelayAttack', 'trafficCongestionSybil', 'zeroSpeedReport']
CACHE = 'results/_pools_cache.pkl'
WINDOW_S = 600.0
FPR_LAT = [0.01, 0.001]


def load_pools():
    if os.path.exists(CACHE):
        with open(CACHE, 'rb') as f:
            pools = pickle.load(f)
        print(f'loaded {len(pools)} subsets from cache', flush=True)
        return pools
    pools = {}
    for atk in ATTACKS:
        name = f'InTAS_highway_2_{atk}'
        t0 = time.time()
        tr = dedup(load_rows(name, 'Train'))
        te = dedup(load_rows(name, 'Test'))
        pools[atk] = (tr, te)
        print(f'loaded {atk:28s} tr={len(tr)} te={len(te)} '
              f'({time.time()-t0:.0f}s)', flush=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(pools, f)
    return pools


def main():
    t0 = time.time()
    pools = load_pools()
    out = {}
    for atk in ATTACKS:
        feats = FEATS_F34 + (F5LEGIT if atk == 'dataReplay' else [])
        tr, te = pools[atk]
        tr_senders = set(tr.sender_id)
        tr = tr[~tr.sender_id.isin(set(te.sender_id))]
        te = te[~te.sender_id.isin(tr_senders)]
        Xtr, ytr = tr[feats].fillna(-1).values, tr['attacker'].values
        Xte, yte = te[feats].fillna(-1).values, te['attacker'].values
        probs = np.zeros(len(te))
        clfs = []
        for seed in range(3):
            clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                         random_state=seed)
            clf.fit(Xtr, ytr)
            clfs.append(clf)
            probs += clf.predict_proba(Xte)[:, 1] / 3.0
        ted = te.copy()
        ted['score'] = probs
        evs = get_events(ted)
        n_ev = len(evs)
        in_ev = np.zeros(len(ted), dtype=bool)
        for a, b in evs:
            in_ev[a:b + 1] = True
        norm = probs[~in_ev]
        res = {'n_test': int(len(ted)), 'n_events': n_ev,
               'msg_auroc': round(float(roc_auc_score(yte, probs)), 4)}

        send_t = ted['sendTime'].values
        for fp in FPR_LAT:
            thr = float(np.quantile(norm, 1.0 - fp)) if len(norm) else np.inf
            det_msgs, det_secs = [], []
            for a, b in evs:
                hit = np.where(probs[a:b + 1] >= thr)[0]
                if len(hit):
                    k = a + int(hit[0])
                    det_msgs.append(int(hit[0]))
                    det_secs.append(float(send_t[k] - send_t[a]) / 1e9)
            ndet = len(det_msgs)
            res[f'lat_fpr_{fp}'] = {
                'thr': round(thr, 4),
                'detection_rate': round(ndet / n_ev, 4) if n_ev else None,
                'median_msgs_to_detect':
                    float(np.median(det_msgs)) if ndet else None,
                'median_secs_to_detect':
                    round(float(np.median(det_secs)), 2) if ndet else None}

        thr01 = float(np.quantile(norm, 0.99)) if len(norm) else np.inf
        rcv = ted['rcvTime'].values.astype(np.int64)
        w = ((rcv - rcv.min()) // int(WINDOW_S * 1e9)).astype(int)
        n_win = int(w.max()) + 1
        auc_ws, fpr_ws, n_skip = [], [], 0
        for wi in range(n_win):
            m = w == wi
            if m.sum() < 50 or len(np.unique(yte[m])) < 2:
                n_skip += 1
                continue
            auc_ws.append(float(roc_auc_score(yte[m], probs[m])))
            nm = m & ~in_ev
            if nm.sum():
                fpr_ws.append(float((probs[nm] >= thr01).mean()))
        res['stability'] = {
            'window_s': WINDOW_S, 'n_windows': n_win,
            'n_valid_auroc': len(auc_ws), 'n_skipped': n_skip,
            'auroc_mean': round(float(np.mean(auc_ws)), 4) if auc_ws else None,
            'auroc_min': round(float(np.min(auc_ws)), 4) if auc_ws else None,
            'auroc_max': round(float(np.max(auc_ws)), 4) if auc_ws else None,
            'fpr01_mean': round(float(np.mean(fpr_ws)), 4) if fpr_ws else None,
            'fpr01_max': round(float(np.max(fpr_ws)), 4) if fpr_ws else None}

        n_t = min(20000, len(Xte))
        Xt = Xte[:n_t]
        for tag, nj in [('njobs1', 1), ('njobs_1', -1)]:
            clfs[0].n_jobs = nj
            ts = []
            for _ in range(3):
                tt = time.time()
                clfs[0].predict_proba(Xt)
                ts.append(time.time() - tt)
            res[f'infer_{tag}_us_per_msg'] = round(min(ts) / n_t * 1e6, 1)
            clfs[0].n_jobs = -1

        out[atk] = res
        print(f'{atk:28s} events={n_ev} det@1%={res["lat_fpr_0.01"]["detection_rate"]} '
              f'med_msgs={res["lat_fpr_0.01"]["median_msgs_to_detect"]} '
              f'winAUC=[{res["stability"]["auroc_min"]},{res["stability"]["auroc_max"]}] '
              f'({time.time()-t0:.0f}s)', flush=True)
        with open('results/operational_metrics.json', 'w') as f:
            json.dump(out, f, indent=2)
    print(f'saved results/operational_metrics.json ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
