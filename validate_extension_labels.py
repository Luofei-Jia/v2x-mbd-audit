"""Standalone validation of the VeReMi Extension class-label mapping.

The published tables for the Extension dataset (Mendeley
10.17632/k62n4z9gdz.1) contradict each other, so the class->attack mapping
used in the paper was adjudicated empirically from per-class kinematic
signatures. This script recomputes every signature family that supported
that adjudication and prints one table per family:

  T1 scale: msg/sender/pseudonym counts, speed stats, move rate, |dpos|
  T2 content: exact-content collision rate (different sender, <=6s),
     zero-speed fraction, speed std, position-jump statistics
  T3 noise-channel sanity: |pos_n - pos| distances (documents that the
     _n columns are zero-filled garbage; attack values live in clean cols)
  T4 within-identity speed variance and speed-position consistency
  T5 trajectory plausibility: distance to nearest normal (class 0) position
  T6 temporal coverage: per-class sendTime range (which classes appear in
     which simulation windows) and messages per real sender (flooding/churn)
  T7 pseudonym churn: pseudonyms per real sender per class

Usage: python3 validate_extension_labels.py [path/to/extension_csv]
"""
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

CSV = sys.argv[1] if len(sys.argv) > 1 else 'data/extension_mixalldata_clean.csv'


def load(cols=None):
    df = pd.read_csv(CSV, usecols=cols)
    df['spd'] = np.hypot(df['spdx'], df['spdy'])
    df = df.sort_values(['senderPseudo', 'sendTime'])
    g = df.groupby('senderPseudo')
    df['dt'] = g['sendTime'].diff()
    df['dpos'] = np.hypot(g['posx'].diff(), g['posy'].diff())
    return df


def t1_scale(df):
    print('T1 scale (per class)')
    print(f'{"cls":>3} {"n_msg":>8} {"n_snd":>6} {"n_psd":>6} {"spd_m":>6} '
          f'{"spd_s":>6} {"rate_m":>7} {"dpos_m":>7}')
    d = df.copy()
    d['rate'] = d['dpos'] / d['dt']
    for c in range(20):
        s = d[d['class'] == c]
        print(f'{c:>3} {len(s):>8} {s["sender"].nunique():>6} '
              f'{s["senderPseudo"].nunique():>6} {s["spd"].mean():>6.2f} '
              f'{s["spd"].std():>6.2f} {s["rate"].median():>7.2f} '
              f'{s["dpos"].mean():>7.2f}')


def t2_content(df):
    df = df.copy()
    df['key'] = (df['posx'].astype(str) + '_' + df['posy'].astype(str) + '_' +
                 df['spdx'].astype(str) + '_' + df['spdy'].astype(str) + '_' +
                 df['aclx'].astype(str) + '_' + df['acly'].astype(str))
    recent = {}
    coll = np.zeros(len(df), dtype=bool)
    times = df['sendTime'].values
    keys = df['key'].values
    snd = df['sender'].values
    for i in range(len(df)):
        t, k, s = times[i], keys[i], snd[i]
        for (tj, sj) in recent.get(k, ()):
            if t - tj <= 6.0 and sj != s:
                coll[i] = True
                break
        lst = recent.setdefault(k, [])
        lst.append((t, s))
        if len(lst) > 8:
            del lst[0]
    df['coll'] = coll
    print('\nT2 content collisions and jumps (per class)')
    print(f'{"cls":>3} {"coll":>6} {"spd0":>6} {"spd_s":>6} '
          f'{"jump>50":>8} {"jump>5":>7}')
    for c in range(20):
        s = df[df['class'] == c]
        print(f'{c:>3} {s["coll"].mean():>6.3f} {(s["spd"] < 0.5).mean():>6.3f} '
              f'{s["spd"].std():>6.2f} {(s["dpos"] > 50).mean():>8.3f} '
              f'{(s["dpos"] > 5).mean():>7.3f}')


def t3_noise(df):
    d = load(['class', 'senderPseudo', 'sendTime', 'posx', 'posy', 'spdx',
              'spdy', 'posx_n', 'posy_n', 'spdx_n', 'spdy_n'])
    d['spd_n'] = np.hypot(d['spdx_n'], d['spdy_n'])
    d['dpn'] = np.hypot(d['posx_n'] - d['posx'], d['posy_n'] - d['posy'])
    d['dsn'] = (d['spd_n'] - d['spd']).abs()
    print('\nT3 noise-channel sanity (per class): _n columns are zero-filled, '
          'attack values live in clean columns')
    print(f'{"cls":>3} {"|posn-pos|":>10} {"|spdn-spd|":>10} {"spd_n_std":>9}')
    for c in range(20):
        s = d[d['class'] == c]
        print(f'{c:>3} {s["dpn"].mean():>10.2f} {s["dsn"].mean():>10.2f} '
              f'{s["spd_n"].std():>9.2f}')


def t4_consistency(df):
    d = df.copy()
    d['rate'] = d['dpos'] / d['dt']
    d['consis'] = (d['rate'] - d['spd']).abs()
    within = d.groupby('senderPseudo')['spd'].std()
    print('\nT4 within-identity speed variance and speed-position consistency')
    print(f'{"cls":>3} {"within_std_p50":>14} {"within_std_p90":>14} '
          f'{"consis_med":>11} {"consis_mean":>12}')
    for c in range(20):
        s = d[d['class'] == c]
        w = within.reindex(s['senderPseudo'].unique())
        print(f'{c:>3} {w.median():>14.3f} {w.quantile(.9):>14.3f} '
              f'{s["consis"].median():>11.3f} {s["consis"].mean():>12.3f}')


def t5_road_distance(df):
    print('\nT5 distance to nearest normal (class 0) position (m)')
    norm = df[df['class'] == 0].sample(100000, random_state=0)[
        ['posx', 'posy']].values
    tree = KDTree(norm)
    print(f'{"cls":>3} {"d_p50":>8} {"d_p90":>8} {"d_mean":>9}')
    for c in range(20):
        s = df[df['class'] == c].sample(min(30000, int((df['class'] == c).sum())),
                                        random_state=0)[['posx', 'posy']].values
        dd, _ = tree.query(s, k=1)
        print(f'{c:>3} {np.median(dd):>8.2f} {np.quantile(dd, .9):>8.2f} '
              f'{dd.mean():>9.2f}')


def t6_coverage(df):
    print('\nT6 temporal coverage and msgs per real sender')
    print(f'{"cls":>3} {"t_min":>10} {"t_max":>12} {"msg_per_sender":>15}')
    for c in range(20):
        s = df[df['class'] == c]
        n_snd = s['sender'].nunique()
        print(f'{c:>3} {s["sendTime"].min():>10.1f} {s["sendTime"].max():>12.1f} '
              f'{len(s) / n_snd:>15.1f}')


def t7_churn(df):
    print('\nT7 pseudonym churn (pseudonyms per real sender)')
    print(f'{"cls":>3} {"psd_per_sender":>14}')
    for c in range(20):
        s = df[df['class'] == c]
        print(f'{c:>3} {s["senderPseudo"].nunique() / s["sender"].nunique():>14.2f}')


def main():
    df = load(['class', 'sendTime', 'sender', 'senderPseudo', 'posx', 'posy',
               'spdx', 'spdy', 'aclx', 'acly'])
    t1_scale(df)
    t2_content(df)
    t3_noise(df)
    t4_consistency(df)
    t5_road_distance(df)
    t6_coverage(df)
    t7_churn(df)


if __name__ == '__main__':
    main()
