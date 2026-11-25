"""Mesures de fiabilité utilisées par les détecteurs professionnels et le banc d'essai RAID."""
import numpy as np


def auroc(y, s):
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    if len(set(y.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def seuil_pour_fpr(y, s, fpr):
    """Plus petit seuil tel que la part de textes humains au-dessus reste <= fpr."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    h = np.sort(s[y == 0])[::-1]
    if len(h) == 0:
        return float("inf")
    k = int(np.floor(fpr * len(h)))
    return float(h[k]) if k < len(h) else float(h[-1])


def tpr_a_fpr(y, s, fpr):
    """Part des textes d'IA repérés quand on n'accepte que `fpr` de faux positifs."""
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    t = seuil_pour_fpr(y, s, fpr)
    ia = s[y == 1]
    return float((ia > t).mean()) if len(ia) else float("nan"), t


def resume(y, s, seuil=None):
    y, s = np.asarray(y), np.asarray(s, dtype=float)
    out = {"n_humains": int((y == 0).sum()), "n_ia": int((y == 1).sum()), "auroc": auroc(y, s)}
    for f in (0.01, 0.05):
        tpr, t = tpr_a_fpr(y, s, f)
        out[f"tpr_a_{int(f * 100)}pc_fpr"] = tpr
        out[f"seuil_{int(f * 100)}pc_fpr"] = t
    if seuil is not None:
        pred = s > seuil
        out["seuil"] = float(seuil)
        out["faux_positifs"] = float(pred[y == 0].mean()) if (y == 0).any() else float("nan")
        out["faux_negatifs"] = float((~pred[y == 1]).mean()) if (y == 1).any() else float("nan")
        out["exactitude"] = float((pred == (y == 1)).mean())
    return out
