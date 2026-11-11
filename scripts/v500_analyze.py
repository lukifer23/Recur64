#!/usr/bin/env python
"""V500 gate analysis. Implements the thresholds frozen in docs/V500_EXPERIMENTS.md V500-E3 and nothing else.

usage: v500_analyze.py <out.json> --val <val-eval.json>... --test <test-eval.json>... --classical <classical-test-eval.json> [--extrap <test-eval.json>...]
Each learned eval file is one training seed; width is chosen ONCE on the pooled VALIDATION runs, never on TEST.
"""
import json, sys, random, statistics

DEEP = range(11, 17)
G3_DEPTHS = range(10, 15)
B = 100_000
CENSOR = 1_000_000
REF = json.load(open('docs/evidence/v500/classical-reference.json'))
CLASSICAL_REF = REF['classical_ref_mean_d11_16']  # 0.3958333...


def load(paths):
    return [json.load(open(p)) for p in paths]


def runs_by_width(ev):
    return {json.dumps(r['width']): r['records'] for r in ev['runs']}


def solved_at(rec, budget=B):
    return bool(rec['solved'] and rec['nodes'] <= budget)


def deep_rate(records, budget=B):
    """mean over depths 11-16 of the per-depth solve rate"""
    rates = []
    for d in DEEP:
        r = [x for x in records if x['depth'] == d]
        if r:
            rates.append(sum(solved_at(x, budget) for x in r) / len(r))
    return sum(rates) / len(rates) if rates else float('nan')


def pooled_indicator(evs, width):
    """per depth: list over positions of the seed-averaged solve indicator (positions are identical across seeds)"""
    per_depth = {d: [] for d in DEEP}
    seeds = [runs_by_width(e)[width] for e in evs]
    for d in DEEP:
        cols = [[x for x in s if x['depth'] == d] for s in seeds]
        n = min(len(c) for c in cols)
        for i in range(n):
            per_depth[d].append(sum(solved_at(c[i]) for c in cols) / len(cols))
    return per_depth


def boot_ci(per_depth, ref, resamples=20000, seed=0x500):
    rng = random.Random(seed)
    stats = []
    depths = [d for d in DEEP if per_depth[d]]
    for _ in range(resamples):
        s = 0.0
        for d in depths:
            v = per_depth[d]
            s += sum(v[rng.randrange(len(v))] for _ in range(len(v))) / len(v)
        stats.append(s / len(depths) - ref)
    stats.sort()
    return stats[int(0.025 * resamples)], stats[int(0.975 * resamples)]


def mean_deep(per_depth):
    return sum(sum(v) / len(v) for v in per_depth.values() if v) / sum(1 for v in per_depth.values() if v)


def choose_width(val_evs):
    best, best_w = -1, None
    for w in runs_by_width(val_evs[0]):
        if json.loads(w) is None:
            continue  # the frozen rule chooses among widths {2,3,4}
        score = sum(deep_rate(runs_by_width(e)[w]) for e in val_evs) / len(val_evs)
        if score > best:
            best, best_w = score, w
    return best_w, best


def g3(test_evs, width, classical):
    """median nodes (unsolved censored at 1e6) learned vs best classical config, depths 10-14"""
    out = {}
    cl = runs_by_width(classical)
    for d in G3_DEPTHS:
        def med(records):
            v = sorted((x['nodes'] if x['solved'] else CENSOR) for x in records if x['depth'] == d)
            return statistics.median(v)
        learned = statistics.mean(med(runs_by_width(e)[width]) for e in test_evs)
        best_cl = min(med(r) for r in cl.values())
        censored = best_cl >= CENSOR
        out[d] = {'learned_median': learned, 'best_classical_median': best_cl, 'ratio': learned / best_cl,
                  'classical_censored': censored, 'pass': censored or learned <= 0.5 * best_cl}
    return out


def main():
    a = sys.argv[1:]
    out = a[0]
    groups = {'--val': [], '--test': [], '--classical': [], '--extrap': []}
    cur = None
    for t in a[1:]:
        if t in groups:
            cur = t
        else:
            groups[cur].append(t)
    val, test = load(groups['--val']), load(groups['--test'])
    classical = load(groups['--classical'])[0]
    width, vscore = choose_width(val)
    rep = {'schema': 'v500_gates_v1', 'classical_reference': CLASSICAL_REF, 'width_chosen_on_val': json.loads(width),
           'val_mean_deep_solve_1e5': vscore}
    per = pooled_indicator(test, width)
    mean = mean_deep(per)
    lo, hi = boot_ci(per, CLASSICAL_REF)
    seed_means = [deep_rate(runs_by_width(e)[width]) for e in test]
    rep['G1'] = {'mean_deep_solve_1e5': mean, 'ci_vs_reference': [lo, hi], 'per_seed': seed_means,
                 'threshold_mean': CLASSICAL_REF + 0.15, 'seed_floor': CLASSICAL_REF + 0.05,
                 'pass': mean >= CLASSICAL_REF + 0.15 and lo > 0 and all(s > CLASSICAL_REF + 0.05 for s in seed_means)}
    rep['by_depth_solve_1e5'] = {d: sum(per[d]) / len(per[d]) for d in DEEP if per[d]}
    rep['G3'] = g3(test, width, classical)
    rep['G3']['pass_all'] = all(v['pass'] for k, v in rep['G3'].items() if k != 'pass_all')
    if groups['--extrap']:
        ex = load(groups['--extrap'])
        ew, _ = choose_width(load([p for p in groups['--val']])) if False else (width, None)
        eper = pooled_indicator(ex, ew)
        emean = mean_deep(eper)
        rep['G2'] = {'extrap_mean_deep_solve_1e5': emean, 'full_mean': mean,
                     'pass': emean >= CLASSICAL_REF and emean >= 0.75 * mean}
    json.dump(rep, open(out, 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
