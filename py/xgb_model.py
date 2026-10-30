import numpy as np, pandas as pd, itertools, sys
from sklearn.model_selection import GroupKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

import os
D = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else '.'
train_df = pd.read_csv(f'{D}/train.csv'); test_df = pd.read_csv(f'{D}/test.csv')
feats = ['probability', 'volatility', 'momentum', 'price_dispersion']  # same 4 predictors as the RF
X_tr, y_tr, g_tr = train_df[feats], train_df['outcome'], train_df['event_group']
X_te, y_te = test_df[feats], test_df['outcome']

def make(p, seed=42):
    return XGBClassifier(n_estimators=p['n'], max_depth=p['d'], learning_rate=p['lr'],
                         min_child_weight=p['mcw'], subsample=0.8, colsample_bytree=1.0,
                         reg_lambda=5.0, eval_metric='logloss', random_state=seed, n_jobs=-1)

# Tune on TRAIN ONLY with grouped CV (never touch test while choosing hyperparameters)
grid = [dict(n=n, d=d, lr=lr, mcw=m) for n, d, lr, m in
        itertools.product([100, 300], [2, 3], [0.03, 0.1], [5, 10])]
gkf = GroupKFold(n_splits=5)
best, best_s = None, 9
for p in grid:
    s = []
    for a, b in gkf.split(X_tr, y_tr, g_tr):
        s.append(brier_score_loss(y_tr.iloc[b], make(p).fit(X_tr.iloc[a], y_tr.iloc[a]).predict_proba(X_tr.iloc[b])[:, 1]))
    if np.mean(s) < best_s: best, best_s = p, np.mean(s)
print('chosen params', best, 'grouped-CV Brier on train', round(best_s, 5))

# Final fit, average over seeds so the number isn't a lucky draw
preds = np.mean([make(best, s).fit(X_tr, y_tr).predict_proba(X_te)[:, 1] for s in range(10)], axis=0)
single = [brier_score_loss(y_te, make(best, s).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]) for s in range(10)]
print('XGBoost test Brier (10-seed avg preds):', round(brier_score_loss(y_te, preds), 5))
print('single-fit mean over 10 seeds:', round(np.mean(single), 5))
print('per-seed range:', round(min(single), 5), '-', round(max(single), 5))
for c in ['probability', 'pred_price', 'pred_full', 'pred_rf']:
    print(f'{c:>12}:', round(brier_score_loss(y_te, test_df[c]), 5))

# Paired bootstrap by event group: XGB vs market
d = (preds - y_te.values) ** 2 - (test_df['probability'].values - y_te.values) ** 2
g = test_df['event_group'].values; ug = np.unique(g); idx = {k: np.where(g == k)[0] for k in ug}
rng = np.random.default_rng(0)
bs = [d[np.concatenate([idx[k] for k in rng.choice(ug, len(ug))])].mean() for _ in range(5000)]
print('XGB - market Brier diff:', round(d.mean(), 5), '95% CI', np.round(np.percentile(bs, [2.5, 97.5]), 5))
test_df['pred_xgb'] = preds
test_df.to_csv('test_with_xgb.csv', index=False)
