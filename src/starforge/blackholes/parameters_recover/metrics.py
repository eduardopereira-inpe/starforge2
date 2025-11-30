import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred)**2)))


def nrmse_by_mean(y_true, y_pred, eps=1e-16):
    return rmse(y_true, y_pred) / (np.mean(np.abs(y_true)) + eps)


def nrmse_by_range(y_true, y_pred):
    return rmse(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + 1e-16)


def huber(y_true, y_pred, delta=1e-6):
    r = y_true - y_pred
    abs_r = np.abs(r)
    mask = abs_r <= delta
    loss = np.empty_like(r)
    loss[mask] = 0.5 * r[mask]**2
    loss[~mask] = delta * (abs_r[~mask] - 0.5 * delta)
    return float(np.mean(loss))


def pearson_corr(y_true, y_pred):
    yt = y_true - np.mean(y_true)
    yp = y_pred - np.mean(y_pred)
    denom = (np.sqrt(np.sum(yt**2)) * np.sqrt(np.sum(yp**2)))
    if denom == 0:
        return 0.0
    return float(np.sum(yt * yp) / denom)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1.0 - ss_res / (ss_tot + 1e-16)


def combined_loss(y_true, y_pred, alpha=0.5):
    # alpha controla peso da amplitude (MAE) vs forma (1 - corr)
    amp = mae(y_true, y_pred)
    form = 1.0 - pearson_corr(y_true, y_pred)  # menor é melhor
    return alpha * amp + (1 - alpha) * form


def mdsape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Symmetric Absolute Percentage Error (in percent)."""
    eps = 1e-13
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    ratio = numerator / (denominator + eps)
    return float(np.median(ratio) * 100.0)


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Log Error."""
    eps = 1e-13
    a = np.log(np.abs(y_true) + eps)
    b = np.log(np.abs(y_pred) + eps)
    return float(np.sqrt(np.mean((a - b)**2)))


def mean_abs_log_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Log-Ratio Error."""
    eps = 1e-13
    ratio = (np.abs(y_pred) + eps) / (np.abs(y_true) + eps)
    return float(np.mean(np.abs(np.log(ratio))))


def relative_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative RMSE normalized by mean absolute true value."""
    eps = 1e-13
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    scale = np.mean(np.abs(y_true)) + eps
    return float(rmse / scale)


def mase(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Scaled Error."""
    eps = 1e-13

    mae_model = np.mean(np.abs(y_true - y_pred))

    if len(y_true) < 2:
        return float(mae_model)

    mae_naive = np.mean(np.abs(np.diff(y_true))) + eps
    return float(mae_model / mae_naive)


def nrmse_percent(y, yhat, denom='mean'):
    rmse = np.sqrt(np.mean((y - yhat)**2))
    if denom == 'mean':
        scale = np.mean(np.abs(y))
    elif denom == 'range':
        scale = np.max(y) - np.min(y)
    else:
        scale = denom  # user-specified
    # proteger contra divisão por zero
    if scale == 0:
        return np.nan
    return rmse / scale * 100


def smape_percent(y, yhat, eps=1e-12):
    # sMAPE = mean( 2|y - yhat| / (|y| + |yhat|) )
    denom = (np.abs(y) + np.abs(yhat)) + eps
    return np.mean(2.0 * np.abs(y - yhat) / denom) * 100


def stabilized_mape_percent(y, yhat, eps=1e-8):
    # evita divisão por números muito pequenos usando max(|y_i|, eps)
    denom = np.maximum(np.abs(y), eps)
    return np.mean(np.abs((y - yhat) / denom)) * 100


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (in percent)."""
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(np.mean(numerator / (denominator + 1e-13)) * 100.0)


def _relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Erro relativo do tipo sMAPE (sem multiplicar por 100).
    Retorna um vetor de erros e_i.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return numerator / (denominator + 1e-13)


def q95_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Percentil 95% dos erros relativos (em percent).
    """
    e = _relative_error(y_true, y_pred)
    return float(np.percentile(e, 95) * 100.0)


def loss_median_q95(y_true: np.ndarray, y_pred: np.ndarray,
                    w_med: float = 0.7, w_q95: float = 0.3) -> float:
    """
    J = w_med * MdSAPE + w_q95 * Q95
    """
    med = mdsape(y_true, y_pred)
    q95 = q95_error(y_true, y_pred)
    return float(w_med * med + w_q95 * q95)
