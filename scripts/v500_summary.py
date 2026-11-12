#!/usr/bin/env python
"""Build docs/evidence/v500/summary.json from the committed evidence (no new measurements)."""
import json, statistics

E = 'docs/evidence/v500/'
SEEDS = (5101, 5102, 5103)


def load(p):
    return json.load(open(p))


def by_width(ev):
    return {json.dumps(r['width']): r['records'] for r in ev['runs']}


def rate(recs, d, bud):
    q = [x for x in recs if x['depth'] == d]
    return sum(x['solved'] and x['nodes'] <= bud for x in q) / len(q)


def med_nodes(recs, d, cens=1_000_000):
    return statistics.median((x['nodes'] if x['solved'] else cens) for x in recs if x['depth'] == d)


out = {'schema': 'v500_summary_v1', 'depths': list(range(1, 17))}
base = {c['config']: c for c in load(E + 'baseline-krvk.json')['configs']}
cl = by_width(load(E + 'eval/classical-test.json'))
full = [by_width(load(E + f'eval/v2-full-{s}-test.json')) for s in SEEDS]
ext = [by_width(load(E + f'eval/v2-extrap-{s}-test.json')) for s in SEEDS]
D = out['depths']

def mean_over(sets, w, bud):
    return [sum(rate(s[w], d, bud) for s in sets) / len(sets) for d in D]

out['solve_1e5'] = {
    'blind_full': [r['solve_1e5'] for r in base['blind_full']['rows']],
    'classical_best': [max(rate(cl[w], d, 100_000) for w in cl) for d in D],
    'learned_w1': mean_over(full, '1', 100_000),
    'learned_w2': mean_over(full, '2', 100_000),
}
out['median_nodes'] = {
    'classical_best': [min(med_nodes(cl[w], d) for w in cl) for d in D],
    'learned_w2': [statistics.mean(med_nodes(s['2'], d) for s in full) for d in D],
    'learned_w1': [statistics.mean(med_nodes(s['1'], d) for s in full) for d in D],
}
out['extrap'] = {
    'w2_1e5': mean_over(ext, '2', 100_000),
    'w3_1e5': mean_over(ext, '3', 100_000),
    'w3_1e6': mean_over(ext, '3', 1_000_000),
    'full_w2_1e5': out['solve_1e5']['learned_w2'],
}
allr = load(E + 'eval/v2-full-5101-test-all.json')['runs'][0]['records']
out['exhaustive_5101'] = {
    'positions': len(allr), 'solved': sum(x['solved'] for x in allr), 'max_nodes': max(x['nodes'] for x in allr),
    'optimal_length_share': sum(x['bound'] == x['depth'] for x in allr if x['bound']) / len(allr),
    'by_depth': {d: sum(1 for x in allr if x['depth'] == d) for d in D},
}
kb = {c['config']: c for c in load(E + 'baseline-kbnvk.json')['configs']}
out['kbn'] = {
    'depths': list(range(1, 34)),
    'baseline_solve_1e6_best': [max(c['rows'][i]['solve_1e6'] for c in kb.values() if i < len(c['rows'])) for i in range(33)],
    'oracle_nodes': {r['depth']: r['median_nodes_solved'] for r in load(E + 'oracle-kbn-w1.json')['rows']},
}
json.dump(out, open(E + 'summary.json', 'w'))
print('wrote summary.json')
print('learned w2 solve@1e5 by depth:', [round(x, 2) for x in out['solve_1e5']['learned_w2']])
print('classical best:', [round(x, 2) for x in out['solve_1e5']['classical_best']])
print('exhaustive:', {k: v for k, v in out['exhaustive_5101'].items() if k != 'by_depth'})
