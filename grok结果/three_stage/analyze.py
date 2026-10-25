import json, os, numpy as np
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 结果文件与脚本同目录
R = json.load(open('results.json')); G = ['G1','G2-GRU','G2-Win','G3','G4-GRU','G4-Win']; S = [0,1,2]; ST = ['overall','motion','slow','dwell']
def get(key, g, st): return R['cv'][key][g][st]
L = []
def P(*a): s = ' '.join(str(x) for x in a); print(s); L.append(s)
P('freeze sha256', R['freeze_sha256'])
P('## CV (5-fold mean of seed-mean macro RMSE; [min,max] over seeds of fold-mean)')
for st in ST:
    P(f'### {st}')
    for g in G:
        fm = np.array([[get(f'fold{f}_seed{s}', g, st)['macro'] for s in S] for f in range(5)])
        jt = np.mean([[get(f'fold{f}_seed{s}', g, st)['joints'] for s in S] for f in range(5)], (0,1))
        sm = fm.mean(0)
        P(f'{g:7s} macro {fm.mean():.4f} [{sm.min():.4f},{sm.max():.4f}] folds', np.round(fm.mean(1),4).tolist(), 'joints', np.round(jt,3).tolist())
P('## paired G4-G2 per fold (seed-mean), overall & slow')
for v in ['GRU','Win']:
    for st in ['overall','slow','dwell']:
        d = np.array([np.mean([get(f'fold{f}_seed{s}', f'G4-{v}', st)['macro']-get(f'fold{f}_seed{s}', f'G2-{v}', st)['macro'] for s in S]) for f in range(5)])
        P(f'{v} {st}: diffs', np.round(d,4).tolist(), f'mean {d.mean():.4f}', 'wins', int((d<0).sum()), '/5', ('-> USEFUL' if st=='overall' and (d<0).all() and d.mean()<0 else ''))
P('## test traj12 (descriptive), seed mean [min,max]')
for st in ST:
    for g in G:
        a = np.array([R['test'][f'seed{s}'][g][st]['macro'] for s in S]); j = np.mean([R['test'][f'seed{s}'][g][st]['joints'] for s in S],0)
        P(f'{st:7s} {g:7s} {a.mean():.4f} [{a.min():.4f},{a.max():.4f}] joints', np.round(j,3).tolist())
open('summary.txt','w').write('\n'.join(L))
