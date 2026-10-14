"""Emit LaTeX table fragments for the 2026-10-14 additions from results JSON
so no number is transcribed by hand (protocol section 10)."""
import json

R = 'results/'
NAMES = {'dosAttack': 'DoS', 'zeroSpeedReport': 'zeroSpeed',
         'constantSpeedOffset': 'constSpeedOff', 'randomSpeedOffset': 'randSpeedOff',
         'randomPositionOffset': 'randPosOff', 'trafficCongestionSybil': 'Sybil',
         'suddenConstantSpeed': 'suddConstSpd', 'reversedHeading': 'revHead',
         'feignedBraking': 'feignBrake', 'suddenStop': 'suddenStop',
         'accelerationMultiplication': 'accelMult', 'dataReplay': 'dataReplay',
         'constantPositionOffset': 'constPosOff', 'positionMirroring': 'posMirror',
         'timeDelayAttack': 'timeDelay'}


def loao_table():
    d = json.load(open(R + 'cross_attack_matrix.json'))
    att = d['attacks']
    mat = [[None if v is None else v for v in row] for row in d['matrix_auroc']]
    rows = []
    for j, a in enumerate(att):
        L = d['loao'][a]
        if L['auroc_mean'] is None:
            rows.append(f"{NAMES[a]} & -- & -- & -- & -- \\\\")
            continue
        others = [(mat[i][j], att[i]) for i in range(len(att)) if i != j
                  and mat[i][j] is not None]
        bo, bsrc = max(others)
        delta = L['auroc_mean'] - bo
        rows.append(
            f"{NAMES[a]} & {mat[j][j]:.3f} & {bo:.3f} "
            f"({NAMES[bsrc]}) & {L['auroc_mean']:.3f} & {delta:+.3f} \\\\")
    body = "\n".join(rows)
    print(f"""\\begin{{table*}}[!t]
\\caption{{Unseen-attack generalization (P2, mean of 3 seeds). \\emph{{Within}}:
detector trained and tested on the same attack (diagonal of
Fig.~\\ref{{fig:matrix}}). \\emph{{Best single}}: strongest transfer from a
single \\emph{{other}} attack (source in parentheses).
\\emph{{LOAO}}: detector trained on normal traffic plus the other 14
attacks.}}
\\label{{tab:loao}}
\\centering
\\begin{{tabular}}{{lcccc}}
\\toprule
Attack & Within & Best single & LOAO & $\\Delta$ (LOAO--best) \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}""")


def operational_table():
    d = json.load(open(R + 'operational_metrics.json'))
    rows = []
    for a in ['dosAttack', 'zeroSpeedReport', 'constantSpeedOffset',
              'randomSpeedOffset', 'randomPositionOffset',
              'trafficCongestionSybil', 'suddenConstantSpeed',
              'reversedHeading', 'feignedBraking', 'suddenStop',
              'accelerationMultiplication', 'dataReplay',
              'constantPositionOffset', 'positionMirroring',
              'timeDelayAttack']:
        r = d[a]
        L1 = r['lat_fpr_0.01']
        L01 = r['lat_fpr_0.001']
        st = r['stability']

        def fmt(x, nd=2):
            return '--' if x is None else f'{x:.{nd}f}'
        rows.append(
            f"{NAMES[a]} & {r['n_events']} & {fmt(L1['detection_rate'])} & "
            f"{fmt(L1['median_msgs_to_detect'], 1)} & "
            f"{fmt(L1['median_secs_to_detect'], 1)} & "
            f"{fmt(L01['detection_rate'])} & "
            f"{fmt(st['auroc_min'], 3)}--{fmt(st['auroc_max'], 3)} & "
            f"{fmt(st['fpr01_max'] * 100 if st['fpr01_max'] is not None else None, 2)} & "
            f"{fmt(r['infer_njobs1_us_per_msg'], 1)} \\\\")
    body = "\n".join(rows)
    print(f"""\\begin{{table*}}[!t]
\\caption{{Operational profile (P2). Events: per-sender attack runs
($\\le$2\\,s gap). Det@FPR: fraction of events whose first message exceeds
the normal-score quantile at 1\\% / 0.1\\% message FPR; med.\\ msg/s:
messages / seconds from event onset to first detection among detected
events. Window AUROC: min--max over 600\\,s test windows; max FPR: largest
per-window false-alarm rate at the global 1\\% threshold. $\\mu$s: RF
inference per message (single core).}}
\\label{{tab:operational}}
\\centering
\\begin{{tabular}}{{lccccccrc}}
\\toprule
Attack & Events & Det@1\\% & Med.\\ msg & Med.\\ s & Det@0.1\\% &
Win.\\ AUROC & Max FPR & $\\mu$s \\\\
\\midrule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table*}}""")


if __name__ == '__main__':
    print('=' * 30, 'LOAO')
    loao_table()
    print('=' * 30, 'OPERATIONAL')
    operational_table()
