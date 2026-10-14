"""Paper figures (color edition):
fig1 audit scatter (colored by inflation delta, collision-free labels),
fig2 difficulty map (sectioned blocks, two-tone: routine vs open),
fig3 timeDelay smoking gun (protocol profile with chance band + overlaid ECDFs)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Patch
from sweep_hard3_f5 import load_rows, dedup

OUT = '.'

plt.rcParams.update({
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6,
    'lines.linewidth': 1.0, 'figure.dpi': 300, 'savefig.dpi': 300,
})

C_TRIV, C_EASY, C_MED, C_HARD = '#4575b4', '#abd9e9', '#fdae61', '#d73027'
C_BLUE, C_RED = '#4575b4', '#d73027'
C_SOFT = '#b8cfe0'


def fig1():
    """Scatter colored by inflation; greedy label de-collision after draw."""
    d = json.load(open('results/sweep_attacks.json'))
    names = {'dosAttack': 'DoS', 'zeroSpeedReport': 'zeroSpeed',
             'constantSpeedOffset': 'constSpeedOff', 'randomSpeedOffset': 'randSpeedOff',
             'randomPositionOffset': 'randPosOff', 'trafficCongestionSybil': 'Sybil',
             'suddenConstantSpeed': 'suddConstSpd', 'reversedHeading': 'revHead',
             'feignedBraking': 'feignBrake', 'suddenStop': 'suddenStop',
             'accelerationMultiplication': 'accelMult', 'dataReplay': 'dataReplay',
             'constantPositionOffset': 'constPosOff', 'positionMirroring': 'posMirror',
             'timeDelayAttack': 'timeDelay'}
    xs, ys, ds, labs = [], [], [], []
    for k, n in names.items():
        p1 = d[k]['P1_F1']['auroc']
        p3 = d[k]['P3_F1']['auroc']
        xs.append(p1); ys.append(p3); ds.append(p3 - p1); labs.append(n)
    xs, ys, ds = map(np.array, (xs, ys, ds))

    fig, ax = plt.subplots(figsize=(3.7, 3.2))
    fig.subplots_adjust(left=0.13, right=0.88, bottom=0.13, top=0.90)
    ax.plot([0.42, 1.06], [0.42, 1.06], '--', color='0.45', lw=0.8, zorder=1)
    sc = ax.scatter(xs, ys, s=22, c=ds, cmap='YlOrRd', vmin=0.0, vmax=0.55,
                    edgecolor='0.25', linewidth=0.4, zorder=3)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.05)
    cb.set_label('inflation  $\\Delta$ = P3$-$P1', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    prefer = {  # staggered downward directions into the empty lower region
        'DoS': (-14, -36), 'timeDelay': (-12, -56), 'revHead': (-8, -14),
        'posMirror': (-20, -32), 'dataReplay': (2, -12), 'Sybil': (18, -36),
        'constSpeedOff': (-30, -52), 'suddConstSpd': (10, -62),
        'randSpeedOff': (14, -22), 'constPosOff': (-4, -34), 'randPosOff': (-14, -40),
        'suddenStop': (-30, -14), 'accelMult': (-24, -44),
        'feignBrake': (-38, -70), 'zeroSpeed': (-14, -64),
    }
    offsets = dict(prefer)
    texts = {}
    for i, l in enumerate(labs):
        texts[l] = ax.annotate(l, (xs[i], ys[i]), textcoords='offset points',
                               xytext=offsets[l], fontsize=5.6, zorder=4)
    ax.set_xlabel('AUROC, honest temporal split (P1, F1)')
    ax.set_ylabel('AUROC, random split (P3, F1)')
    ax.set_xlim(0.42, 1.06)
    ax.set_ylim(0.42, 1.06)
    ax.set_title('Memorization audit: random vs temporal split', fontsize=8)

    # diagonal annotation, rotated to the on-screen slope of the diagonal
    p1c = ax.transData.transform((0.55, 0.55))
    p2c = ax.transData.transform((0.75, 0.75))
    ang = np.degrees(np.arctan2(p2c[1] - p1c[1], p2c[0] - p1c[0]))
    diag_txt = ax.text(0.56, 0.545, 'no inflation', fontsize=6, color='0.35',
                       rotation=ang, rotation_mode='anchor', zorder=2)

    # deterministic label placement: preferred offset first, then an angle sweep
    # at growing radius; first candidate clear of other labels, marker dots, the
    # diagonal note and the axes frame wins. Hardest points placed first.
    from matplotlib.transforms import Bbox
    fig.canvas.draw()
    ren = fig.canvas.get_renderer()
    ab = ax.bbox
    pad = 1.5
    pts_px = [ax.transData.transform((x, y)) for x, y in zip(xs, ys)]
    mhalf = 11  # half-size of a scatter marker in px (s=22 at 300 dpi)
    diag_box = diag_txt.get_window_extent(ren).expanded(2, 2)

    K = fig.dpi / 72.0  # offset points -> px
    anch, size = {}, {}
    for l in labs:
        bb = texts[l].get_window_extent(ren)
        px, py = pts_px[labs.index(l)]
        anch[l] = (bb.x0 - px - offsets[l][0] * K, bb.y0 - py - offsets[l][1] * K)
        size[l] = (bb.width, bb.height)

    def box_for(l, dx, dy):  # dx, dy in px
        px, py = pts_px[labs.index(l)]
        return Bbox.from_bounds(px + dx + anch[l][0], py + dy + anch[l][1],
                                size[l][0], size[l][1])

    def nearmark(l):
        i = labs.index(l)
        return sum(1 for j in range(len(labs)) if j != i and
                   np.hypot((xs[i] - xs[j]) / 0.64, (ys[i] - ys[j]) / 0.64) < 0.18)

    def candidates_ok(l, dxp, dyp, obstacles):
        bx = box_for(l, dxp, dyp).expanded(pad, pad)
        if bx.x0 < ab.x0 or bx.x1 > ab.x1 or bx.y0 < ab.y0 or bx.y1 > ab.y1:
            return False
        if bx.overlaps(diag_box) or any(bx.overlaps(ob) for ob in obstacles):
            return False
        return not any(
            labs[j] != l and bx.overlaps(
                Bbox.from_bounds(mx - mhalf, my - mhalf, 2 * mhalf, 2 * mhalf))
            for j, (mx, my) in enumerate(pts_px))

    def search(l, obstacles):
        if candidates_ok(l, offsets[l][0] * K, offsets[l][1] * K, obstacles):
            return offsets[l]
        dx0, dy0 = prefer[l]
        if dx0 == 0 and dy0 == 0:
            dx0, dy0 = 0, -20
        r0 = max(np.hypot(dx0, dy0) * K, 14 * K)
        a0 = np.arctan2(dy0, dx0)
        for scale in (1.0, 1.2, 1.45, 1.75):
            r = min(r0 * scale, 90)  # stay near the point
            for dd in (0, 8, -8, 16, -16, 24, -24, 32, -32, 40, -40, 50, -50,
                       60, -60, 72, -72, 85, -85, 100, -100, 115, -115, 130,
                       -130, 145, -145, 160, -160, 180):
                a = a0 + np.radians(dd)
                if candidates_ok(l, r * np.cos(a), r * np.sin(a), obstacles):
                    return (r * np.cos(a) / K, r * np.sin(a) / K)
        return None

    # iterative repair: each round re-searches only labels currently in
    # conflict, hardest first; boxes of already-repositioned labels are
    # updated in place so later labels in the same round avoid them too
    hard = sorted(labs, key=lambda l: (-nearmark(l), labs.index(l)))
    for _ in range(20):
        cur = {l: box_for(l, offsets[l][0] * K, offsets[l][1] * K).expanded(pad, pad)
               for l in labs}
        bad = set()
        for a in range(len(labs)):
            for b in range(a + 1, len(labs)):
                if cur[labs[a]].overlaps(cur[labs[b]]):
                    bad.update((labs[a], labs[b]))
        for l in labs:
            bx = cur[l]
            if (bx.x0 < ab.x0 or bx.x1 > ab.x1 or bx.y0 < ab.y0
                    or bx.y1 > ab.y1 or bx.overlaps(diag_box)):
                bad.add(l)
            for j, (mx, my) in enumerate(pts_px):
                if labs[j] != l and bx.overlaps(
                        Bbox.from_bounds(mx - mhalf, my - mhalf,
                                         2 * mhalf, 2 * mhalf)):
                    bad.add(l)
        if not bad:
            break
        moved = False
        for l in [h for h in hard if h in bad]:
            obstacles = [cur[m] for m in labs if m != l]
            new = search(l, obstacles)
            if new is not None and new != offsets[l]:
                offsets[l] = new
                cur[l] = box_for(l, new[0] * K, new[1] * K).expanded(pad, pad)
                moved = True
        if not moved:
            break
    for l in labs:
        texts[l].set_position(offsets[l])

    # report residual label-label / label-marker / label-diagonal / out-of-axes
    fig.canvas.draw()
    boxes = {l: texts[l].get_window_extent(ren) for l in labs}
    n_ov = sum(1 for a in range(len(labs)) for b in range(a + 1, len(labs))
               if boxes[labs[a]].overlaps(boxes[labs[b]]))
    n_mk = 0
    for j, (mx, my) in enumerate(pts_px):
        mb = Bbox.from_bounds(mx - mhalf, my - mhalf, 2 * mhalf, 2 * mhalf)
        n_mk += sum(1 for l in labs if labs.index(l) != j and boxes[l].overlaps(mb))
    n_dg = sum(1 for l in labs if boxes[l].overlaps(diag_box))
    n_oob = sum(1 for l in labs
                if boxes[l].x0 < ab.x0 or boxes[l].x1 > ab.x1
                or boxes[l].y0 < ab.y0 or boxes[l].y1 > ab.y1)
    print(f'fig1 labels: overlaps={n_ov}, on-marker={n_mk}, on-diag={n_dg}, '
          f'out-of-axes={n_oob}')
    if n_ov:
        print('  overlapping pairs:',
              [(labs[a], labs[b]) for a in range(len(labs))
               for b in range(a + 1, len(labs))
               if boxes[labs[a]].overlaps(boxes[labs[b]])])
    if n_oob:
        print('  out-of-axes labels:',
              [l for l in labs if boxes[l].x0 < ab.x0 or boxes[l].x1 > ab.x1
               or boxes[l].y0 < ab.y0 or boxes[l].y1 > ab.y1])
    fig.savefig(f'{OUT}/fig1_audit.png')
    plt.close(fig)


def fig2():
    d = json.load(open('results/sweep_attacks.json'))
    f4 = json.load(open('results/sweep_f4.json'))
    f5 = json.load(open('results/sweep_hard3_f5.json'))
    vals = [
        ('DoS', d['dosAttack']['P2_F3']['auroc'], C_TRIV),
        ('zeroSpeed', d['zeroSpeedReport']['P2_F3']['auroc'], C_TRIV),
        ('constSpeedOff', d['constantSpeedOffset']['P2_F3']['auroc'], C_TRIV),
        ('randSpeedOff', d['randomSpeedOffset']['P2_F3']['auroc'], C_TRIV),
        ('randPosOff', d['randomPositionOffset']['P2_F3']['auroc'], C_TRIV),
        ('Sybil', d['trafficCongestionSybil']['P2_F3']['auroc'], C_TRIV),
        ('suddConstSpd', d['suddenConstantSpeed']['P2_F3']['auroc'], C_EASY),
        ('revHead', f4['reversedHeading']['P2']['auroc'], C_EASY),
        ('feignBrake', d['feignedBraking']['P2_F3']['auroc'], C_EASY),
        ('suddenStop', d['suddenStop']['P2_F3']['auroc'], C_EASY),
        ('accelMult', d['accelerationMultiplication']['P2_F3']['auroc'], C_EASY),
        ('dataReplay', f5['dataReplay']['P2_F34+F5legit']['auroc'], C_EASY),
        ('constPosOff', d['constantPositionOffset']['P2_F3']['auroc'], C_MED),
        ('posMirror', f5['positionMirroring']['P2_F34+F5legit']['auroc'], C_HARD),
        ('timeDelay', f5['timeDelayAttack']['P2_F34+F5legit']['auroc'], C_HARD),
    ]
    vals.sort(key=lambda t: -t[1])
    gap = 1.5
    spans = [6, 6, 1, 2]
    heads = ['trivially detectable', 'detectable', 'medium', 'hard / open']
    ys, hy, cur = [], [], gap
    for n in spans:
        hy.append(cur - 0.8)
        for _ in range(n):
            ys.append(cur)
            cur += 1
        cur += gap
    ys = np.array(ys)

    fig, ax = plt.subplots(figsize=(3.7, 3.5), constrained_layout=True)
    ax.axvline(0.5, color='0.2', lw=0.9, ls=':')
    cols = [C_SOFT if v[2] in (C_TRIV, C_EASY) else v[2] for v in vals]
    ax.barh(ys, [v[1] for v in vals], height=0.62, color=cols,
            edgecolor='0.2', lw=0.4)
    for yi, v in zip(ys, vals):
        ax.text(v[1] + 0.008, yi, f'{v[1]:.3f}', va='center', fontsize=5.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([v[0] for v in vals])
    for yh, h in zip(hy, heads):
        ax.text(0.402, yh, h, fontsize=5.8, color='0.3', style='italic',
                path_effects=[pe.withStroke(linewidth=2.2, foreground='white')])
    ax.text(0.505, hy[0] - 0.6, 'chance', fontsize=6.0, color='0.2')
    ax.set_xlim(0.4, 1.06)
    ax.set_xlabel('message AUROC (P2, best legitimate features)')
    ax.invert_yaxis()
    ax.set_title('Honest difficulty map: 15 attacks', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    fig.savefig(f'{OUT}/fig2_map.png', bbox_inches='tight')
    plt.close(fig)


def fig3():
    d = json.load(open('results/sweep_attacks.json'))
    f5 = json.load(open('results/sweep_hard3_f5.json'))
    seq = json.load(open('results/seq_baselines.json'))
    gnn = json.load(open('results/stgnn_baseline.json'))
    rows = [  # left-to-right
        ('P3-F1', d['timeDelayAttack']['P3_F1']['auroc'], C_RED),
        ('P1-F1', d['timeDelayAttack']['P1_F1']['auroc'], C_SOFT),
        ('P1-F34', f5['timeDelayAttack']['P1_F34']['auroc'], C_SOFT),
        ('P2-F34', f5['timeDelayAttack']['P2_F34']['auroc'], C_SOFT),
        ('LSTM', seq['timeDelayAttack']['LSTM']['auroc'], C_SOFT),
        ('Transformer', seq['timeDelayAttack']['Transformer']['auroc'], C_SOFT),
        ('ST-GNN', gnn['timeDelayAttack']['auroc'], C_SOFT),
    ]
    vals = [r[1] for r in rows]
    honest = np.array(vals[1:])
    lo, hi = honest.min(), honest.max()
    xs = np.array([0.0, 1.7, 2.6, 3.5, 4.4, 5.3, 6.2])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.6),
                                   gridspec_kw={'width_ratios': [1.02, 1]},
                                   constrained_layout=True)
    ax1.axhspan(lo, hi, color='0.87', lw=0)
    ax1.axhline(0.5, color='0.2', lw=0.9, ls=':')
    ax1.plot([0, 0], [hi + 0.005, vals[0] - 0.025], color=C_RED, lw=1.0,
             zorder=2)
    ax1.plot(xs[1:], vals[1:], '-', color='0.45', lw=1.0, zorder=2)
    ax1.plot(xs[1:], vals[1:], 'o', ms=4, mfc=C_SOFT, mec='0.2', mew=0.4,
             ls='none', zorder=3)
    ax1.plot([xs[0]], [vals[0]], 'D', ms=6, mfc=C_RED, mec='0.2', mew=0.4,
             ls='none', zorder=3)
    for i, (x, v) in enumerate(zip(xs, vals)):
        lab = f'{v:.3f}' if i == 0 else f'{v:.2f}'
        ax1.text(x, v + 0.03, lab, ha='center', fontsize=6)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([r[0] for r in rows], rotation=18, ha='right',
                        fontsize=6.5)
    stroke = [pe.withStroke(linewidth=2.2, foreground='white')]
    ax1.text(0.42, 0.80, 'memorization-eligible split', fontsize=6.2,
             color='0.3', style='italic', path_effects=stroke)
    ax1.text(3.95, 0.60, 'honest protocols and model classes', fontsize=6.2,
             color='0.3', style='italic', path_effects=stroke, ha='center')
    ax1.text(-0.62, 0.505, 'chance', fontsize=6, color='0.25', va='bottom',
             path_effects=stroke)
    ax1.set_xlim(-0.75, 6.9)
    ax1.set_ylim(0.40, 1.10)
    ax1.set_ylabel('AUROC')
    ax1.set_title('(a) timeDelay: protocols and models', fontsize=8)
    ax1.spines[['top', 'right']].set_visible(False)

    df = dedup(load_rows('InTAS_highway_2_timeDelayAttack', 'Test'))
    att = df[df.attacker == 1]['delay_ms'].values
    nor = df[df.attacker == 0]['delay_ms'].values
    for data, c, ls, lab in ((nor, C_BLUE, '-', 'normal'),
                             (att, C_RED, (0, (3, 2)), 'attacker')):
        xs = np.sort(data)
        yy = np.arange(1, len(xs) + 1) / len(xs)
        ax2.step(xs, yy, where='post', color=c, ls=ls, lw=1.3, label=lab)
    ax2.set_xlim(0.94, 2.2)
    ax2.set_ylim(0, 1.02)
    ax2.set_xlabel('message delay (ms)')
    ax2.set_ylabel('ECDF')
    ax2.legend(frameon=False, loc='upper left')
    ax2.set_title('(b) empirical CDFs', fontsize=8)
    qa = np.quantile(att, [0.25, 0.5, 0.75])
    qn = np.quantile(nor, [0.25, 0.5, 0.75])
    ax2.text(0.42, 0.40,
             'near-identical ECDFs\n'
             f'quartiles (att) {qa[0]:.2f}/{qa[1]:.2f}/{qa[2]:.2f}\n'
             f'(nor) {qn[0]:.2f}/{qn[1]:.2f}/{qn[2]:.2f}\n'
             'delay-alone AUROC 0.51',
             transform=ax2.transAxes, fontsize=6.2)
    ax2.spines[['top', 'right']].set_visible(False)
    fig.savefig(f'{OUT}/fig3_delay.png', bbox_inches='tight')
    plt.close(fig)


def fig4():
    """Cross-attack transfer matrix (rows=train attack, cols=test attack) +
    LOAO column; diagonal boxed; Blues scale consistent with honest-protocol
    color language of fig3."""
    d = json.load(open('results/cross_attack_matrix.json'))
    att = d['attacks']
    M = np.array([[np.nan if v is None else v for v in row]
                  for row in d['matrix_auroc']], dtype=float)
    loao = np.array([d['loao'][a]['auroc_mean'] if d['loao'][a]['auroc_mean']
                     is not None else np.nan for a in att])
    names = {'dosAttack': 'DoS', 'zeroSpeedReport': 'zeroSpeed',
             'constantSpeedOffset': 'constSpeedOff', 'randomSpeedOffset': 'randSpeedOff',
             'randomPositionOffset': 'randPosOff', 'trafficCongestionSybil': 'Sybil',
             'suddenConstantSpeed': 'suddConstSpd', 'reversedHeading': 'revHead',
             'feignedBraking': 'feignBrake', 'suddenStop': 'suddenStop',
             'accelerationMultiplication': 'accelMult', 'dataReplay': 'dataReplay',
             'constantPositionOffset': 'constPosOff', 'positionMirroring': 'posMirror',
             'timeDelayAttack': 'timeDelay'}
    labs = [names[a] for a in att]
    n = len(att)

    fig = plt.figure(figsize=(7.2, 3.95))
    gs = fig.add_gridspec(2, 3, height_ratios=[15, 0.75],
                          width_ratios=[15, 0.55, 1.15],
                          hspace=1.25, wspace=0.04,
                          left=0.115, right=0.985, top=0.875, bottom=0.155)
    ax = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 2])
    cax = fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap('Blues').copy()
    cmap.set_bad('0.88')
    norm = matplotlib.colors.Normalize(vmin=0.45, vmax=1.0)

    im = ax.imshow(M, cmap=cmap, norm=norm, aspect='auto')
    for i in range(n):
        v = M[i, i]
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                                   ec='k', lw=1.0, zorder=5))
        if not np.isnan(v):
            ax.text(i, i, f'{v:.2f}', ha='center', va='center', fontsize=4.4,
                    color='white' if norm(v) > 0.62 else 'k', zorder=6)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labs, rotation=90, fontsize=6.2)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labs, fontsize=6.2)
    for t in ax.get_xticklabels():
        if t.get_text() in ('posMirror', 'timeDelay'):
            t.set_color(C_HARD)
            t.set_fontweight('bold')
    for t in ax.get_yticklabels():
        if t.get_text() in ('posMirror', 'timeDelay'):
            t.set_color(C_HARD)
            t.set_fontweight('bold')
    ax.set_xlabel('evaluated on attack (column)', fontsize=7)
    ax.set_ylabel('trained on attack (row)', fontsize=7)
    for x in np.arange(-0.5, n, 1):
        ax.axvline(x, color='w', lw=0.2)
        ax.axhline(x, color='w', lw=0.2)
    ax.set_title('Cross-attack transfer (P2, message AUROC; diag. boxed)',
                 fontsize=8)

    ax2.imshow(loao.reshape(-1, 1), cmap=cmap, norm=norm, aspect='auto')
    for i, v in enumerate(loao):
        if not np.isnan(v):
            ax2.text(0, i, f'{v:.2f}', ha='center', va='center', fontsize=4.6,
                     color='white' if norm(v) > 0.62 else 'k')
    ax2.set_xticks([0])
    ax2.set_xticklabels(['LOAO'], rotation=90, fontsize=6.2, fontweight='bold')
    ax2.set_yticks([])
    ax2.set_title('unseen', fontsize=6.5)

    cb = fig.colorbar(im, cax=cax, orientation='horizontal')
    cb.set_label('AUROC', fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    fig.savefig(f'{OUT}/fig4_matrix.png', dpi=300)
    plt.close(fig)
    n_nan = int(np.isnan(M).sum())
    print(f'fig4 saved (missing cells: {n_nan})')


if __name__ == '__main__':
    fig1()
    fig2()
    fig3()
    fig4()
    print('figures saved')
