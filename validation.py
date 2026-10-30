# Validation for the C161 Polymarket project. Needs train.csv + test.csv (with pred_* columns) in the working folder.
import numpy as np, pandas as pd, os, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

D = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else '.'
tr = pd.read_csv(f'{D}/train.csv'); te = pd.read_csv(f'{D}/test.csv')
full = pd.concat([tr, te], ignore_index=True)
F4 = ['probability', 'volatility', 'momentum', 'price_dispersion']
FL = ['logit_market', 'volatility_pp', 'momentum_pp', 'dispersion_pp']   # what the R logistic used
XP = dict(n_estimators=100, max_depth=2, learning_rate=0.03, min_child_weight=5, subsample=0.8, reg_lambda=5.0)
brier = lambda p, y: float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))

def fit_predict(name, A, B):
    """Fit model `name` on dataframe A, predict probabilities for dataframe B."""
    if name == 'Polymarket':   return B['probability'].values
    if name == 'Logistic (price)': return LogisticRegression(C=1e9, max_iter=2000).fit(A[['logit_market']], A.outcome).predict_proba(B[['logit_market']])[:, 1]
    if name == 'Logistic (full)':  return LogisticRegression(C=1e9, max_iter=2000).fit(A[FL], A.outcome).predict_proba(B[FL])[:, 1]
    if name == 'Random Forest':    return np.mean([RandomForestClassifier(500, random_state=s, n_jobs=-1).fit(A[F4], A.outcome).predict_proba(B[F4])[:, 1] for s in range(5)], axis=0)
    if name == 'XGBoost':          return np.mean([XGBClassifier(**XP, eval_metric='logloss', random_state=s, n_jobs=-1).fit(A[F4], A.outcome).predict_proba(B[F4])[:, 1] for s in range(5)], axis=0)
MODELS = ['Polymarket', 'Logistic (price)', 'Logistic (full)', 'Random Forest', 'XGBoost']

def boot_ci(d, groups, n=5000, seed=0):
    rng = np.random.default_rng(seed); ug = np.unique(groups); idx = {k: np.where(groups == k)[0] for k in ug}
    b = [d[np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])].mean() for _ in range(n)]
    return d.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)

def table(P, y, g, title):
    print(f'\n=== {title} (n={len(y)}, event groups={len(np.unique(g))}) ===')
    print(f"{'model':<18}{'Brier':>9}{'  diff vs market  [95% CI, grouped bootstrap]':>50}")
    base = (P['Polymarket'] - y) ** 2
    for m in MODELS:
        d = (P[m] - y) ** 2 - base
        if m == 'Polymarket': print(f'{m:<18}{brier(P[m], y):>9.5f}'); continue
        mean, lo, hi = boot_ci(d, g)
        sig = '  <-- CI excludes 0' if (lo > 0 or hi < 0) else ''
        print(f'{m:<18}{brier(P[m], y):>9.5f}{mean:>+14.5f}  [{lo:+.5f}, {hi:+.5f}]{sig}')

# 1) Held-out test split: fit on train, score on test (same split as the slides)
y, g = te.outcome.values, te.event_group.values
P_te = {m: fit_predict(m, tr, te) for m in MODELS}
table(P_te, y, g, 'TEST SPLIT (Python refits; RF/logistic may differ slightly from the R slides)')

# 2) 10-fold event-grouped CV on all 1,094 markets: far less dependent on one lucky split
rng = np.random.default_rng(1); ug = np.unique(full.event_group); rng.shuffle(ug)
fold = {k: i % 10 for i, k in enumerate(ug)}; f = full.event_group.map(fold).values
oof = {m: np.zeros(len(full)) for m in MODELS}
for k in range(10):
    A, B = full[f != k], full[f == k]
    for m in MODELS: oof[m][f == k] = fit_predict(m, A, B)
table(oof, full.outcome.values, full.event_group.values, '10-FOLD EVENT-GROUPED CV (all markets)')

# 3) Calibration plot (test split): predicted vs observed YES rate, equal-count bins
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot([0, 1], [0, 1], 'k--', lw=1, label='perfect')
for m, c in [('Polymarket', 'tab:blue'), ('Random Forest', 'tab:green'), ('XGBoost', 'tab:orange')]:
    p = pd.Series(P_te[m]); q = pd.qcut(p.rank(method='first'), 8, labels=False)
    t = pd.DataFrame({'p': p, 'y': y, 'q': q}).groupby('q').mean()
    ax[0].plot(t.p, t.y, 'o-', color=c, label=m)
ax[0].set(xlabel='Predicted YES probability', ylabel='Observed YES rate', title='Calibration, test split (8 equal-count bins)'); ax[0].legend()
lo_ = full.assign(b=pd.cut(full.probability, [0, .02, .1, .3, .5, .7, .9, 1], include_lowest=True)).groupby('b', observed=True).agg(n=('outcome', 'size'), fc=('probability', 'mean'), obs=('outcome', 'mean'))
x = np.arange(len(lo_)); ax[1].bar(x - .2, lo_.fc, .4, label='Market price'); ax[1].bar(x + .2, lo_.obs, .4, label='Observed YES')
ax[1].set_xticks(x); ax[1].set_xticklabels([f'{i}\nn={n}' for i, n in zip(lo_.index.astype(str), lo_.n)], fontsize=7)
ax[1].set(title='Market calibration, all 1,094 markets', ylabel='Probability'); ax[1].legend()
plt.tight_layout(); plt.savefig('calibration.png', dpi=160); print('\nsaved calibration.png')
