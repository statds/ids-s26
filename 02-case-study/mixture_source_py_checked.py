from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


@dataclass
class MixtureControl:
    tol: float = 1e-6
    maxit: int = 200
    sign_cons: bool = False
    mu_cons: bool = False
    sigma_type: str = "diag"
    iter_inner: int = 1
    update_gamma_first: bool = False
    warm_start: bool = False
    trace: bool = False


@dataclass
class MixtureResult:
    Pik: np.ndarray
    Gamma: np.ndarray
    Alpha: np.ndarray
    Mu: np.ndarray
    Sigma: np.ndarray
    Pi: np.ndarray
    diff: List[float]
    loglike: List[float]
    obj: List[float]
    df: float
    aic: float
    bic: float
    nout: int


def _pinv(a: np.ndarray) -> np.ndarray:
    return np.linalg.pinv(a)


def _safe_diag_cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] <= 1:
        v = np.nanvar(x, axis=0)
        v = np.where(np.isfinite(v) & (v > 1e-6), v, 1.0)
        return np.diag(v)
    c = np.cov(x, rowvar=False)
    c = np.atleast_2d(c)
    return np.diag(np.clip(np.diag(c), 1e-6, None))


def _safe_full_cov(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if x.shape[0] <= 1:
        v = np.nanvar(x, axis=0)
        v = np.where(np.isfinite(v) & (v > 1e-6), v, 1.0)
        return np.diag(v)
    c = np.cov(x, rowvar=False)
    c = np.atleast_2d(c)
    c = c + np.eye(c.shape[0]) * 1e-6
    return c


def _matrix_sqrt_inv(sigma: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.clip(vals, 1e-10, None)
    return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T


def initial_values(Y: np.ndarray, random_state: int = 0) -> Dict[str, np.ndarray]:
    """Python version of the R initialisation logic for the 3-component model.

    The R path function uses a template with component centers at 0, -3, +3 and
    then maps the fitted KMeans centers to reference / low / high by nearest-center
    matching. This function mirrors that design more closely than a generic ordering.
    """
    P, M, S = Y.shape
    # When a replicate is completely missing, np.nanmean emits a warning.
    # Silence it here and replace any resulting NaNs with 0 (as in the R version).
    with np.errstate(invalid="ignore"):
        alpha_ini = np.nanmean(Y, axis=(0, 1))
    alpha_ini = np.nan_to_num(alpha_ini, nan=0.0)
    with np.errstate(invalid="ignore"):
        Y_avg = np.nanmean(Y - alpha_ini.reshape(1, 1, S), axis=2)
    Y_avg = np.nan_to_num(Y_avg, nan=0.0)

    Mu_template = np.column_stack([
        np.zeros(M),
        -3.0 * np.ones(M),
        3.0 * np.ones(M),
    ])

    km = KMeans(n_clusters=3, n_init=20, random_state=random_state)
    labels = km.fit_predict(Y_avg)
    centers = km.cluster_centers_

    dists = np.sum((centers[:, None, :] - Mu_template.T[None, :, :]) ** 2, axis=2)
    cluster1id = int(np.argmin(dists[:, 0]))
    cluster2id = int(np.argmin(dists[:, 1]))
    cluster3id = int(np.argmin(dists[:, 2]))

    # Fallback if assignments collide.
    if len({cluster1id, cluster2id, cluster3id}) < 3:
        order = np.argsort(centers.mean(axis=1))
        cluster2id, cluster1id, cluster3id = int(order[0]), int(order[1]), int(order[2])

    Pi = np.array([
        np.mean(labels == cluster1id),
        np.mean(labels == cluster2id),
        np.mean(labels == cluster3id),
    ], dtype=float)
    Pi = np.clip(Pi, 1e-6, None)
    Pi /= Pi.sum()

    Mu = np.zeros((M, 3), dtype=float)
    Mu[:, 1] = centers[cluster2id]
    Mu[:, 2] = centers[cluster3id]

    Sigma = np.zeros((M, M, 3), dtype=float)
    Sigma[:, :, 0] = _safe_full_cov(Y_avg[labels == cluster1id])
    Sigma[:, :, 1] = _safe_diag_cov(Y_avg[labels == cluster2id])
    Sigma[:, :, 2] = _safe_diag_cov(Y_avg[labels == cluster3id])

    Gamma = np.zeros((P, M), dtype=float)
    return {"Pi": Pi, "Mu": Mu, "Sigma": Sigma, "Alpha": alpha_ini, "Gamma": Gamma}


def _observed_ids(Y: np.ndarray, s: int) -> np.ndarray:
    # Return indices of proteins with at least one observed value in replicate s.
    # Avoid warnings from reductions over all-NaN rows.
    return np.where(~np.all(np.isnan(Y[:, :, s]), axis=1))[0]


def _e_step(Y: np.ndarray, Pi: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray, Alpha: np.ndarray, Gamma: np.ndarray):
    P, M, S = Y.shape
    K = 3
    log_d = np.tile(np.log(np.clip(Pi, 1e-300, None)), (P, 1))

    for s in range(S):
        ids = _observed_ids(Y, s)
        if len(ids) == 0:
            continue
        Ys = Y[ids, :, s]

        mvn1 = multivariate_normal(mean=Alpha[s] + Mu[:, 0], cov=Sigma[:, :, 0], allow_singular=True)
        log_d[ids, 0] += mvn1.logpdf(Ys - Gamma[ids, :])

        for k in (1, 2):
            mvnk = multivariate_normal(mean=Alpha[s] + Mu[:, k], cov=Sigma[:, :, k], allow_singular=True)
            log_d[ids, k] += mvnk.logpdf(Ys)

    lse = logsumexp(log_d, axis=1, keepdims=True)
    Pik = np.exp(log_d - lse)
    return Pik, float(np.sum(lse))


def _update_alpha(Y: np.ndarray, Pik: np.ndarray, Mu: np.ndarray, Sigma: np.ndarray, Alpha: np.ndarray) -> np.ndarray:
    P, M, S = Y.shape
    Alpha_new = Alpha.copy()
    for s in range(S):
        ids = _observed_ids(Y, s)
        if len(ids) == 0:
            continue

        Ys = Y[ids, :, s]
        sum_ik = np.zeros((len(ids), 3), dtype=float)
        sinv_k = np.ones(3, dtype=float)

        for k in (1, 2):
            Sigmakinv = _pinv(Sigma[:, :, k])
            Yids2 = Ys - Mu[:, k]
            sum_ik[:, k] = np.sum(Sigmakinv @ Yids2.T, axis=0)
            sinv_k[k] = np.sum(Sigmakinv)

        Sigma1inv = _pinv(Sigma[:, :, 0])
        sum_ik[:, 0] = np.sum(Sigma1inv @ Ys.T, axis=0)
        sinv_k[0] = np.sum(Sigma1inv)

        numer = np.sum(Pik[ids, :] * sum_ik)
        denom = np.sum((Pik[ids, :] * sinv_k).sum(axis=1))
        Alpha_new[s] = numer / max(denom, 1e-300)
    return Alpha_new


def _constrained_mu_diag(mu: np.ndarray, sign: str) -> np.ndarray:
    if sign == "nonpos":
        return np.minimum(mu, 0.0)
    if sign == "nonneg":
        return np.maximum(mu, 0.0)
    raise ValueError("sign must be 'nonpos' or 'nonneg'")


def _update_mu(Y: np.ndarray, Pik: np.ndarray, Sigma: np.ndarray, Alpha: np.ndarray, Mu: np.ndarray,
               sign_cons: bool = False, mu_cons: bool = False) -> np.ndarray:
    P, M, S = Y.shape
    Sigma2inv = _pinv(Sigma[:, :, 1])
    Sigma3inv = _pinv(Sigma[:, :, 2])
    Mu_new = Mu.copy()

    if mu_cons:
        sum_low = np.zeros((M, S), dtype=float)
        sum_high = np.zeros((M, S), dtype=float)
        divisor = np.zeros((M, M), dtype=float)
        for s in range(S):
            ids = _observed_ids(Y, s)
            if len(ids) == 0:
                continue
            Ys = Y[ids, :, s]
            sum_low[:, s] = np.sum((Ys - Alpha[s]) @ Sigma2inv * Pik[ids, 1][:, None], axis=0)
            sum_high[:, s] = np.sum((Ys - Alpha[s]) @ Sigma3inv * Pik[ids, 2][:, None], axis=0)
            divisor += np.sum(Pik[ids, 1]) * Sigma2inv + np.sum(Pik[ids, 2]) * Sigma3inv
        Mu_new[:, 1] = _pinv(divisor) @ np.sum(sum_low - sum_high, axis=1)
        Mu_new[:, 2] = -Mu_new[:, 1]
    else:
        sum_low = np.zeros((M, S), dtype=float)
        sum_high = np.zeros((M, S), dtype=float)
        divisor_low = np.zeros((M, M), dtype=float)
        divisor_high = np.zeros((M, M), dtype=float)
        for s in range(S):
            ids = _observed_ids(Y, s)
            if len(ids) == 0:
                continue
            Ys = Y[ids, :, s]
            sum_low[:, s] = np.sum((Ys - Alpha[s]) @ Sigma2inv * Pik[ids, 1][:, None], axis=0)
            sum_high[:, s] = np.sum((Ys - Alpha[s]) @ Sigma3inv * Pik[ids, 2][:, None], axis=0)
            divisor_low += np.sum(Pik[ids, 1]) * Sigma2inv
            divisor_high += np.sum(Pik[ids, 2]) * Sigma3inv

        mu2 = _pinv(divisor_low) @ np.sum(sum_low, axis=1)
        mu3 = _pinv(divisor_high) @ np.sum(sum_high, axis=1)
        if sign_cons:
            # Exact orthant projection for the implemented diagonal-covariance branch.
            mu2 = _constrained_mu_diag(mu2, "nonpos")
            mu3 = _constrained_mu_diag(mu3, "nonneg")
        Mu_new[:, 1] = mu2
        Mu_new[:, 2] = mu3

    Mu_new[:, 0] = 0.0
    return Mu_new


def _update_sigma_diag(Y: np.ndarray, Pik: np.ndarray, Gamma: np.ndarray, Alpha: np.ndarray, Mu: np.ndarray) -> np.ndarray:
    P, M, S = Y.shape
    Sigma = np.zeros((M, M, 3), dtype=float)

    for k in (1, 2):
        sum_s = np.zeros((M, S), dtype=float)
        divisor = 0.0
        for s in range(S):
            ids = _observed_ids(Y, s)
            if len(ids) == 0:
                continue
            Ys = Y[ids, :, s]
            Yids2 = (Ys - Alpha[s] - Mu[:, k]) ** 2
            sum_s[:, s] = np.sum(Yids2 * Pik[ids, k][:, None], axis=0)
            divisor += np.sum(Pik[ids, k])
        vals = np.sum(sum_s, axis=1) / max(divisor, 1e-300)
        Sigma[:, :, k] = np.diag(np.clip(vals, 1e-10, None))

    sum_ref = np.zeros((M, M), dtype=float)
    divisor = 0.0
    for s in range(S):
        ids = _observed_ids(Y, s)
        if len(ids) == 0:
            continue
        ss = (Y[ids, :, s] - Gamma[ids, :] - Alpha[s]) * np.sqrt(Pik[ids, 0])[:, None]
        sum_ref += ss.T @ ss
        divisor += np.sum(Pik[ids, 0])
    Sigma[:, :, 0] = sum_ref / max(divisor, 1e-300)
    Sigma[:, :, 0] += np.eye(M) * 1e-10
    return Sigma


def _update_gamma_ghard(Y: np.ndarray, Pik: np.ndarray, Sigma1: np.ndarray, Alpha: np.ndarray, lam: float) -> np.ndarray:
    P, M, S = Y.shape
    Sig1sqrtinv = _matrix_sqrt_inv(Sigma1)
    Ytemp = np.empty_like(Y)
    for s in range(S):
        # Some Y entries may be NaN (missing replicates); this would trigger
        # a RuntimeWarning during subtraction, so we silence it here.
        with np.errstate(invalid="ignore"):
            Ytemp[:, :, s] = Y[:, :, s] - Alpha[s]
    with np.errstate(invalid="ignore"):
        GrpY0 = np.nanmean(Ytemp, axis=2).T
    GrpY = Sig1sqrtinv @ GrpY0

    # Count the number of replicates containing at least one observed value per protein.
    # Avoid warnings by not reducing across all-NaN slices.
    srepnum = np.sum(~np.all(np.isnan(Y), axis=1), axis=1)
    srepnum = np.clip(srepnum, 1, None)
    l2rowsq = np.sum(GrpY ** 2, axis=0)
    l2lambda = (lam ** 2) / np.clip(Pik[:, 0], 1e-300, None) / srepnum

    Gamma = np.zeros((P, M), dtype=float)
    keep = l2rowsq > l2lambda
    Gamma[keep, :] = GrpY0[:, keep].T
    return Gamma


def mixture_penalized(Y: np.ndarray, lam: float, ini: Dict[str, np.ndarray] | None = None,
                      method: str = "ghard", control: MixtureControl | None = None) -> MixtureResult:
    if method != "ghard":
        raise NotImplementedError("This checked Python version implements only method='ghard'.")
    if control is None:
        control = MixtureControl()
    if control.sigma_type != "diag":
        raise NotImplementedError("This checked Python version implements only sigma_type='diag'.")
    if ini is None:
        ini = initial_values(Y)

    P, M, S = Y.shape
    Pi = ini["Pi"].copy()
    Mu = ini["Mu"].copy()
    Sigma = ini["Sigma"].copy()
    Alpha = ini["Alpha"].copy()
    Gamma = ini["Gamma"].copy()

    theta = np.concatenate([Pi.ravel(), Mu.ravel(), Sigma.ravel(), Alpha.ravel(), Gamma.ravel()])
    diff = [control.tol * 2]
    obj_hist: List[float] = []
    loglike_hist: List[float] = []
    it = 0

    while it < control.maxit and diff[-1] > control.tol:
        theta0 = theta.copy()
        Pik, loglike = _e_step(Y, Pi, Mu, Sigma, Alpha, Gamma)
        nout = int(np.sum(np.sum(np.abs(Gamma), axis=1) != 0))
        obj = float(loglike - 0.5 * (lam ** 2) * nout)
        loglike_hist.append(loglike)
        obj_hist.append(obj)

        Pi = np.sum(Pik, axis=0) / P

        if control.update_gamma_first and it == 0:
            Gamma = _update_gamma_ghard(Y, Pik, Sigma[:, :, 0], Alpha, lam)

        for _ in range(control.iter_inner):
            Alpha = _update_alpha(Y, Pik, Mu, Sigma, Alpha)
            Mu = _update_mu(Y, Pik, Sigma, Alpha, Mu, sign_cons=control.sign_cons, mu_cons=control.mu_cons)
            Sigma = _update_sigma_diag(Y, Pik, Gamma, Alpha, Mu)
            Gamma = _update_gamma_ghard(Y, Pik, Sigma[:, :, 0], Alpha, lam)

        theta = np.concatenate([Pi.ravel(), Mu.ravel(), Sigma.ravel(), Alpha.ravel(), Gamma.ravel()])
        denom = max(np.sum(theta0 ** 2), 1e-300)
        diff.append(float(np.sum((theta - theta0) ** 2) / denom))
        it += 1
        if control.trace:
            print(f"diff = {diff[-1]:.4f} loglike = {loglike:.4f} obj = {obj:.4f} nout = {nout}")

    Pik, loglike = _e_step(Y, Pi, Mu, Sigma, Alpha, Gamma)
    nout = int(np.sum(np.sum(np.abs(Gamma), axis=1) != 0))
    obj = float(loglike - 0.5 * (lam ** 2) * nout)
    loglike_hist.append(loglike)
    obj_hist.append(obj)

    Ytemp = np.empty_like(Y)
    for s in range(S):
        with np.errstate(invalid="ignore"):
            Ytemp[:, :, s] = Y[:, :, s] - Alpha[s]
    with np.errstate(invalid="ignore"):
        GrpY0 = np.nanmean(Ytemp, axis=2).T
    gamma_norm = np.sqrt(np.sum(Gamma ** 2, axis=1))
    grp_norm = np.sqrt(np.sum(GrpY0 ** 2, axis=0))
    frac = np.divide(gamma_norm, np.clip(grp_norm, 1e-300, None))

    if control.mu_cons:
        df1 = S + M + (M * (M - 1) / 2 + M + M) + 2
    else:
        # Matches the R expression for the implemented diagonal branch.
        # Sigma1 is full symmetric; Sigma2 and Sigma3 are diagonal.
        df1 = S + 2 * M + (M * (M - 1) / 2 + M + M) + 2
    df2 = nout + np.sum(frac) * (M - 1)

    aic = -2 * loglike + 2 * df1 + 2 * df2 * P * M / max(P * M - df2 - 1, 1)
    bic = -2 * loglike + np.log(P * M) * df1 + np.log(P * M) * df2 * P * M / max(P * M - df2 - 1, 1)

    return MixtureResult(
        Pik=Pik,
        Gamma=Gamma,
        Alpha=Alpha,
        Mu=Mu,
        Sigma=Sigma,
        Pi=Pi,
        diff=diff,
        loglike=loglike_hist,
        obj=obj_hist,
        df=float(df1 + df2),
        aic=float(aic),
        bic=float(bic),
        nout=nout,
    )


def mixture_penalized_path(Y: np.ndarray, lambdas: np.ndarray, ini: Dict[str, np.ndarray] | None = None,
                           method: str = "ghard", control: MixtureControl | None = None):
    if control is None:
        control = MixtureControl(warm_start=True)
    current_ini = ini
    fits = []
    for lam in np.asarray(lambdas, dtype=float):
        fit = mixture_penalized(Y, lam=lam, ini=current_ini, method=method, control=control)
        fits.append(fit)
        if control.warm_start:
            current_ini = {
                "Pi": fit.Pi.copy(),
                "Mu": fit.Mu.copy(),
                "Sigma": fit.Sigma.copy(),
                "Alpha": fit.Alpha.copy(),
                "Gamma": fit.Gamma.copy(),
            }
    return fits
