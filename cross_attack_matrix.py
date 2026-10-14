"""Cross-attack transfer matrix + leave-one-attack-out (protocol 9l).

D_i: RF trained on attack i's P2 train pool (tr2_i = Train minus Test senders),
positive = attacker messages, uniform features F34+F5legit (29).
Pairwise sender disjointness: evaluated on te2_j minus senders(tr2_i),
where te2_j = Test minus Train senders (sweep convention).
LOAO_j: train = concat of tr2_i for all i!=j (duplicate normal messages
across subsets dropped), test above with union of train senders removed.
"""
import json, os, pickle, time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sweep_hard3_f5 import load_rows, dedup, FEATS_F34, F5LEGIT

ATTACKS = ['accelerationMultiplication', 'constantPositionOffset',
           'constantSpeedOffset', 'dataReplay', 'dosAttack', 'feignedBraking',
           'positionMirroring', 'randomPositionOffset', 'randomSpeedOffset',
           'reversedHeading', 'suddenConstantSpeed', 'suddenStop',
           'timeDelayAttack', 'trafficCongestionSybil', 'zeroSpeedReport']
FEATS = FEATS_F34 + F5LEGIT
CACHE = 'results/_pools_cache.pkl'


def load_all():
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
    os.makedirs('results', exist_ok=True)
    with open(CACHE, 'wb') as f:
        pickle.dump(pools, f)
    print(f'cache saved to {CACHE}', flush=True)
    return pools


def train_clfs(Xtr, ytr, n_seeds=3):
    clfs = []
    for seed in range(n_seeds):
        clf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                     random_state=seed)
        clf.fit(Xtr, ytr)
        clfs.append(clf)
    return clfs


def eval_clfs(clfs, Xte, yte):
    aucs, aprs = [], []
    for clf in clfs:
        p = clf.predict_proba(Xte)[:, 1]
        aucs.append(float(roc_auc_score(yte, p)))
        aprs.append(float(average_precision_score(yte, p)))
    return aucs, aprs


def main():
    t0 = time.time()
    pools = load_all()

    tr2, te2, TR = {}, {}, {}
    for atk in ATTACKS:
        tr, te = pools[atk]
        tr2[atk] = tr[~tr.sender_id.isin(set(te.sender_id))]
        te2[atk] = te[~te.sender_id.isin(set(tr.sender_id))]
        TR[atk] = set(tr2[atk].sender_id)

    nA = len(ATTACKS)
    M_auc = np.full((nA, nA), np.nan)
    M_apr = np.full((nA, nA), np.nan)
    detail = {}
    for i, src in enumerate(ATTACKS):
        tr_i = tr2[src]
        clfs = train_clfs(tr_i[FEATS].fillna(-1).values,
                          tr_i['attacker'].values)
        print(f'[{i}] trained {src} ({time.time()-t0:.0f}s)', flush=True)
        for j, tgt in enumerate(ATTACKS):
            ev = te2[tgt][~te2[tgt].sender_id.isin(TR[src])]
            n_pos = int(ev['attacker'].sum())
            n_neg = int(len(ev) - n_pos)
            if n_pos < 10 or n_neg < 10:
                print(f'  NA -> {tgt} (pos={n_pos} neg={n_neg})', flush=True)
                continue
            aucs, aprs = eval_clfs(clfs, ev[FEATS].fillna(-1).values,
                                   ev['attacker'].values)
            M_auc[i, j] = float(np.mean(aucs))
            M_apr[i, j] = float(np.mean(aprs))
            detail[f'{src}|{tgt}'] = {
                'auroc_mean': round(M_auc[i, j], 4),
                'auroc_seeds': [round(v, 4) for v in aucs],
                'auprc_mean': round(M_apr[i, j], 4),
                'n_pos': n_pos, 'n_neg': n_neg}
        row = ' '.join(f'{M_auc[i, j]:.2f}' if not np.isnan(M_auc[i, j]) else ' na '
                       for j in range(nA))
        print(f'row {i:2d} {src[:26]:26s} {row}  ({time.time()-t0:.0f}s)',
              flush=True)

    print('\n=== LOAO ===', flush=True)
    loao = {}
    for j, held in enumerate(ATTACKS):
        parts = [tr2[src] for src in ATTACKS if src != held]
        big = pd.concat(parts, ignore_index=True)
        n0 = len(big)
        neg_mask = big['attacker'] == 0
        neg = big[neg_mask].drop_duplicates(['sender_id', 'sendTime'],
                                            keep='first')
        big = pd.concat([neg, big[~neg_mask]], ignore_index=True)
        dropped = n0 - len(big)
        others = set().union(*[TR[s] for s in ATTACKS if s != held])
        te_lo = te2[held][~te2[held].sender_id.isin(others)]
        n_pos = int(te_lo['attacker'].sum())
        n_neg = int(len(te_lo) - n_pos)
        res = {'n_train': len(big), 'n_dup_neg_dropped': int(dropped),
               'n_pos': n_pos, 'n_neg': n_neg}
        if n_pos < 10 or n_neg < 10:
            res['auroc_mean'] = None
        else:
            clfs = train_clfs(big[FEATS].fillna(-1).values,
                              big['attacker'].values)
            aucs, aprs = eval_clfs(clfs, te_lo[FEATS].fillna(-1).values,
                                   te_lo['attacker'].values)
            col = M_auc[:, j]
            best_i = int(np.nanargmax(col)) if not np.all(np.isnan(col)) else -1
            diag = M_auc[ATTACKS.index(held), j]
            res.update({
                'auroc_mean': round(float(np.mean(aucs)), 4),
                'auroc_seeds': [round(v, 4) for v in aucs],
                'auprc_mean': round(float(np.mean(aprs)), 4),
                'best_single': round(float(np.nanmax(col)), 4) if best_i >= 0 else None,
                'best_single_src': ATTACKS[best_i] if best_i >= 0 else None,
                'diag_within': None if np.isnan(diag) else round(float(diag), 4)})
        loao[held] = res
        print(f'LOAO {held:28s} auroc={res.get("auroc_mean")} '
              f'(best_single={res.get("best_single")}, '
              f'diag={res.get("diag_within")})  ({time.time()-t0:.0f}s)',
              flush=True)

    out = {
        'meta': {
            'scenario': 'highway_2',
            'features': FEATS,
            'rf': '200 trees, seeds 0-2, per-seed AUROC averaged',
            'protocol': ('D_i trained on tr2_i = Train minus Test senders; '
                         'evaluated on te2_j minus senders(tr2_i); '
                         'te2_j = Test minus Train senders; LOAO train = '
                         'concat tr2_i (i!=j), duplicate normal messages by '
                         '(sender_id, sendTime) dropped'),
        },
        'attacks': ATTACKS,
        'matrix_auroc': [[None if np.isnan(v) else round(float(v), 4)
                          for v in row] for row in M_auc],
        'matrix_auprc': [[None if np.isnan(v) else round(float(v), 4)
                          for v in row] for row in M_apr],
        'detail': detail,
        'loao': loao,
    }
    with open('results/cross_attack_matrix.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f'saved results/cross_attack_matrix.json ({time.time()-t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    main()
