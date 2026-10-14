"""Cross-dataset axis (protocol 9j): VeReMi Extension <-> NextGen highway_2.

Extension has no receiver fields -> aligned sender-side F3-lite features
(spd, acl, hed, dt, rate, consistency, move, dir_consistency), computed
per observable identity (Extension senderPseudo / NextGen sender_id).
RF 200x3 seeds, message-level AUROC+AUPRC.

E1  Extension internal baseline (temporal split 70/30 by sendTime, ~P1).
E2  NextGen highway_2 Train -> Extension test (attack-family alignment).
E3  Extension train -> NextGen highway_2 Test (reverse).
E4  NextGen timeDelay -> all 19 Extension classes (chance expected) +
    E4b locator: Extension {6,10,11,13} -> NextGen {constSpeedOff,
    dataReplay, timeDelay} to pin down delayedMessages.

Training negatives subsampled to 150k (RandomState(42)); test uses all
negatives (unbiased AUPRC at true prevalence).
"""
import json, os, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sweep_hard3_f5 import load_rows, dedup

ROOT = '.'
EXT_CSV = f'{ROOT}/data/extension_mixalldata_clean.csv'
CACHE = f'{ROOT}/data/extension_feats.npz'
ALIGNED = ['spd', 'acl', 'hed', 'dt', 'rate', 'consistency', 'move', 'dir_cons']
NEG_TRAIN = 150000
SPLIT_T = 0.7 * 86400.0

PAIRS = [
    ('dosAttack', [13, 14, 15]),
    ('dataReplay', [10, 11, 17]),
    ('constantPositionOffset', [2]),
    ('constantSpeedOffset', [6]),
    ('randomPositionOffset', [4]),
    ('randomSpeedOffset', [8]),
    ('zeroSpeedReport', [9]),
    ('trafficCongestionSybil', [19, 12]),
]
DELAY_CANDS = [6, 10, 11, 13]
E4B_TARGETS = ['constantSpeedOffset', 'dataReplay', 'timeDelayAttack']


def load_extension():
    """Return (cls, sendTime, X) with X = ALIGNED features (NaN for first msg)."""
    if os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        return d['cls'], d['sendTime'], d['X']
    t0 = time.time()
    df = pd.read_csv(EXT_CSV, usecols=['class', 'sendTime', 'senderPseudo',
                                       'posx', 'posy', 'spdx', 'spdy',
                                       'aclx', 'acly', 'hedx', 'hedy'])
    df['spd'] = np.hypot(df['spdx'], df['spdy'])
    df['acl'] = np.hypot(df['aclx'], df['acly'])
    df['hed'] = np.degrees(np.arctan2(df['hedy'], df['hedx']))
    df = df.sort_values(['senderPseudo', 'sendTime'])
    g = df.groupby('senderPseudo', sort=False)
    dt = g['sendTime'].diff().values
    dx = g['posx'].diff().values
    dy = g['posy'].diff().values
    dist = np.hypot(dx, dy)
    df['dt'] = dt
    df['rate'] = np.where(dt > 0, 1.0 / np.where(dt > 0, dt, np.nan), np.nan)
    df['consistency'] = np.abs(dist / dt - df['spd'])
    df['move'] = dist
    df['dir_cons'] = np.abs(
        (np.degrees(np.arctan2(dy, dx)) - df['hed'] + 180) % 360 - 180)
    X = df[ALIGNED].values.astype(np.float32)
    cls = df['class'].values
    st = df['sendTime'].values
    np.savez(CACHE, cls=cls, sendTime=st, X=X)
    print(f'extension features cached ({time.time()-t0:.0f}s)', flush=True)
    return cls, st, X


def ext_task(cls, st, X, c, split):
    """Rows of class c in window split ('train'|'test'), negatives subsampled."""
    win = st < SPLIT_T if split == 'train' else st >= SPLIT_T
    pos = (cls == c) & win
    neg = (cls == 0) & win
    n_neg = NEG_TRAIN if split == 'train' else -1
    rng = np.random.RandomState(42)
    if n_neg > 0 and neg.sum() > n_neg:
        neg = neg.copy()
        idx = np.where(neg)[0]
        neg[idx[rng.choice(len(idx), len(idx) - n_neg, replace=False)]] = False
    m = pos | neg
    return np.nan_to_num(X[m], nan=-1.0, posinf=0.0, neginf=0.0), \
        (pos[m]).astype(np.float32)


def fit_eval_parts(Xtr, ytr, Xte, yte):
    aucs, aprs = [], []
    for seed in range(3):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(roc_auc_score(yte, p))
        aprs.append(average_precision_score(yte, p))
    return float(np.mean(aucs)), float(np.mean(aprs))


def nextgen_task(atk, split):
    """NextGen attack subset (full split, no P2 filter; cross-dataset is
    sender-disjoint by construction) as (X, y) on ALIGNED columns."""
    name = f'InTAS_highway_2_{atk}'
    df = dedup(load_rows(name, split))
    ng_cols = ['s_spd', 's_acl', 's_hed', 'dt_s', 'rate', 'consistency', 'move',
               'dir_consistency']
    X = np.nan_to_num(df[ng_cols].fillna(-1).values.astype(np.float32),
                      nan=-1.0, posinf=0.0, neginf=0.0)
    y = df['attacker'].values.astype(np.float32)
    return X, y


def main():
    t0 = time.time()
    cls, st, X = load_extension()
    out = {'note': 'Extension temporal split at sendTime<60480s (70%), '
                   'train negatives subsampled to 150k (RandomState(42)), '
                   'test on all negatives; aligned sender-side features '
                   'spd/acl/hed/dt/rate/consistency/move/dir_cons',
           'E1_extension_baseline': {}, 'E2_nextgen_to_extension': {},
           'E3_extension_to_nextgen': {}, 'E4_timeDelay_matrix': {},
           'E4b_locator': {}}

    # ---- E1: Extension internal temporal baseline ----
    print('=== E1 Extension internal baseline ===', flush=True)
    for c in range(1, 20):
        Xtr, ytr = ext_task(cls, st, X, c, 'train')
        Xte, yte = ext_task(cls, st, X, c, 'test')
        if yte.sum() == 0 or ytr.sum() == 0:
            out['E1_extension_baseline'][str(c)] = {'auroc': None, 'auprc': None,
                                                    'n_pos_test': int(yte.sum())}
            print(f'  cls {c}: no positives in a window, skipped', flush=True)
            continue
        auc, apr = fit_eval_parts(Xtr, ytr, Xte, yte)
        out['E1_extension_baseline'][str(c)] = {'auroc': round(auc, 4),
                                                'auprc': round(apr, 4),
                                                'n_pos_test': int(yte.sum())}
        print(f'  cls {c:>2}: auc={auc:.3f} apr={apr:.3f}', flush=True)

    # ---- E2: NextGen Train -> Extension test ----
    print('=== E2 NextGen -> Extension ===', flush=True)
    ng_cache = {}
    for atk, cs in PAIRS + [('timeDelayAttack', DELAY_CANDS)]:
        print(f'  E2 {atk} ...', flush=True)
        if atk not in ng_cache:
            ng_cache[atk] = nextgen_task(atk, 'Train')
        Xtr, ytr = ng_cache[atk]
        out['E2_nextgen_to_extension'][atk] = {}
        for c in cs:
            Xte, yte = ext_task(cls, st, X, c, 'test')
            auc, apr = fit_eval_parts(Xtr, ytr, Xte, yte)
            out['E2_nextgen_to_extension'][atk][str(c)] = {
                'auroc': round(auc, 4), 'auprc': round(apr, 4)}
            print(f'    -> ext cls {c:>2}: auc={auc:.3f} apr={apr:.3f}', flush=True)

    # ---- E3: Extension train -> NextGen Test ----
    print('=== E3 Extension -> NextGen ===', flush=True)
    for atk, cs in PAIRS:
        print(f'  E3 {atk} ...', flush=True)
        Xte, yte = nextgen_task(atk, 'Test')
        out['E3_extension_to_nextgen'][atk] = {}
        for c in cs:
            Xtr, ytr = ext_task(cls, st, X, c, 'train')
            auc, apr = fit_eval_parts(Xtr, ytr, Xte, yte)
            out['E3_extension_to_nextgen'][atk][str(c)] = {
                'auroc': round(auc, 4), 'auprc': round(apr, 4)}
            print(f'    ext cls {c:>2} -> {atk}: auc={auc:.3f} apr={apr:.3f}',
                  flush=True)

    # ---- E4: timeDelay -> all Extension classes (expect chance) ----
    print('=== E4 timeDelay matrix ===', flush=True)
    Xtr, ytr = ng_cache.get('timeDelayAttack', nextgen_task('timeDelayAttack',
                                                            'Train'))
    for c in range(1, 20):
        Xte, yte = ext_task(cls, st, X, c, 'test')
        auc, apr = fit_eval_parts(Xtr, ytr, Xte, yte)
        out['E4_timeDelay_matrix'][str(c)] = {'auroc': round(auc, 4),
                                              'auprc': round(apr, 4)}
        print(f'  timeDelay -> ext cls {c:>2}: auc={auc:.3f}', flush=True)

    # ---- E4b: locator, Extension {6,10,11,13} -> NextGen targets ----
    print('=== E4b locator ===', flush=True)
    ng_targets = {t: nextgen_task(t, 'Test') for t in E4B_TARGETS}
    for c in DELAY_CANDS:
        Xtr, ytr = ext_task(cls, st, X, c, 'train')
        out['E4b_locator'][str(c)] = {}
        for t in E4B_TARGETS:
            Xte, yte = ng_targets[t]
            auc, apr = fit_eval_parts(Xtr, ytr, Xte, yte)
            out['E4b_locator'][str(c)][t] = {'auroc': round(auc, 4),
                                             'auprc': round(apr, 4)}
            print(f'  ext cls {c:>2} -> {t}: auc={auc:.3f}', flush=True)

    os.makedirs('results', exist_ok=True)
    with open('results/cross_dataset_extension.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/cross_dataset_extension.json ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
