from typing import Dict, List

import numpy as np


def init_summary_table() -> Dict:
    return {
        'local': {
            'losses': [],
            'metrics': {
                'w1': [],
                'kl': [],
                'l1': [],
                'l2': [],
                'cvar': [],
                'regret': [],
                'acc': []
            }
        },
        'fedavg': {
            'losses': [],
            'metrics': {
                'w1': [],
                'kl': [],
                'l1': [],
                'l2': [],
                'cvar': [],
                'regret': [],
                'acc': []
            }
        },
        'fedtrunc': {
            'losses': [],
            'metrics': {
                'w1': [],
                'kl': [],
                'l1': [],
                'l2': [],
                'cvar': [],
                'regret': [],
                'acc': []
            }
        },
        'fedrl': {
            'losses': [],
            'kl_losses': [],
            'metrics': {
                'w1': [],
                'kl': [],
                'l1': [],
                'l2': [],
                'cvar': [],
                'regret': [],
                'acc': []
            }
        },
    }


def compute_metrics(pred_hist, truth_hist, z, cvar_alpha=0.1):
    W1s, KLs, L1s, L2s, CVARs = [], [], [], [], []
    for a in range(pred_hist.shape[0]):
        p = pred_hist[a]; q = truth_hist[a]
        cdf_p = np.cumsum(p); cdf_q = np.cumsum(q)
        dz     = z[1] - z[0]
        W1s.append(np.sum(np.abs(cdf_p - cdf_q))*dz)

        eps=1e-12
        KLs.append(np.sum(q * (np.log(q+eps)-np.log(p+eps))))

        mean_p=np.dot(p,z); mean_q=np.dot(q,z)
        L1s.append(abs(mean_p-mean_q))
        L2s.append((mean_p-mean_q)**2)

        # left tail CVaR
        cdf = np.cumsum(p)
        w = np.diff(np.concatenate(([0.0], np.minimum(cdf, cvar_alpha))))
        cvar_p = (w*z).sum()/(cvar_alpha+1e-12)
        cdf = np.cumsum(q)
        w = np.diff(np.concatenate(([0.0], np.minimum(cdf, cvar_alpha))))
        cvar_q = (w*z).sum()/(cvar_alpha+1e-12)
        CVARs.append(abs(cvar_p-cvar_q))

    # regret and acc
    mean_true = truth_hist.dot(z)
    best_true = mean_true.argmax()
    best_pred = (pred_hist.dot(z)).argmax()
    regret=float(mean_true[best_true]-mean_true[best_pred])
    acc=float(best_true==best_pred)

    return {
        'w1': np.mean(W1s),
        'kl': np.mean(KLs),
        'l1': np.mean(L1s),
        'l2': np.mean(L2s),
        'cvar':np.mean(CVARs),
        'regret': regret,
        'acc': acc
    }


def aggregate_summaries(summaries: List[Dict]) -> Dict:
    methods = summaries[0].keys()
    agg = {}

    for method in methods:
        agg[method] = {
            "losses": {},
            "metrics": {}
        }

        # aggregate losses
        losses = [np.array(s[method]["losses"]) for s in summaries]
        max_len = max(len(l) for l in losses)
        losses = [np.pad(l, (0, max_len-len(l)), constant_values=np.nan) for l in losses]
        losses = np.stack(losses, axis=0)  # shape [n_seeds, T]
        agg[method]["losses"]["mean"] = np.nanmean(losses, axis=0)
        agg[method]["losses"]["std"]  = np.nanstd(losses, axis=0)

        # aggregate metrics
        for metric in summaries[0][method]["metrics"].keys():
            vals = [np.array(s[method]["metrics"][metric]) for s in summaries]
            max_len = max(len(v) for v in vals)
            vals = [np.pad(v, (0, max_len-len(v)), constant_values=np.nan) for v in vals]
            vals = np.stack(vals, axis=0)
            agg[method]["metrics"][metric] = {
                "mean": np.nanmean(vals, axis=0),
                "std":  np.nanstd(vals, axis=0),
            }

    return agg