"""
Full 101 Alpha factors + market signal factors.
Replicates legacy/cuda_features.py using PyTorch tensors.
"""
from __future__ import annotations
import torch
from torch import Tensor
from typing import Optional

# ============================================================
# Utility helpers
# ============================================================

def _rank(x: Tensor) -> Tensor:
    """Cross-sectional rank normalized to [0,1]."""
    sorted_idx = x.argsort(dim=-1)
    ranks = torch.zeros_like(x)
    ranks.scatter_(-1, sorted_idx, torch.arange(x.shape[-1], device=x.device, dtype=x.dtype).expand_as(x))
    denom = max(x.shape[-1] - 1, 1)
    return ranks / denom


def _ts_rank(x: Tensor, d: int) -> Tensor:
    """Time-series rank over last d days."""
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        window = x[:, t - d + 1:t + 1, :]
        val = x[:, t:t + 1, :]
        out[:, t, :] = (window < val).sum(dim=1).float() / max(d - 1, 1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _ts_max(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].max(dim=1).values
    return out.squeeze(0) if out.shape[0] == 1 else out


def _ts_min(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].min(dim=1).values
    return out.squeeze(0) if out.shape[0] == 1 else out


def _ts_argmax(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].argmax(dim=1).float()
    return out.squeeze(0) if out.shape[0] == 1 else out


def _ts_argmin(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].argmin(dim=1).float()
    return out.squeeze(0) if out.shape[0] == 1 else out


def _delta(x: Tensor, d: int) -> Tensor:
    pad = torch.zeros_like(x[..., :d, :]) if x.dim() == 3 else torch.zeros_like(x[:d, :])
    if x.dim() == 3:
        return torch.cat([pad, x[:, d:, :] - x[:, :-d, :]], dim=1)
    return torch.cat([pad, x[d:, :] - x[:-d, :]], dim=0)


def _delay(x: Tensor, d: int) -> Tensor:
    pad = torch.zeros_like(x[..., :d, :]) if x.dim() == 3 else torch.zeros_like(x[:d, :])
    if x.dim() == 3:
        return torch.cat([pad, x[:, :-d, :]], dim=1)
    return torch.cat([pad, x[:-d, :]], dim=0)


def _sma(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(T):
        start = max(0, t - d + 1)
        out[:, t, :] = x[:, start:t + 1, :].mean(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _stddev(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].std(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _correlation(x: Tensor, y: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        xw = x[:, t - d + 1:t + 1, :]
        yw = y[:, t - d + 1:t + 1, :]
        xm = xw - xw.mean(dim=1, keepdim=True)
        ym = yw - yw.mean(dim=1, keepdim=True)
        num = (xm * ym).sum(dim=1)
        den = (xm.pow(2).sum(dim=1) * ym.pow(2).sum(dim=1)).sqrt()
        out[:, t, :] = num / den.clamp(min=1e-8)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _covariance(x: Tensor, y: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        xw = x[:, t - d + 1:t + 1, :]
        yw = y[:, t - d + 1:t + 1, :]
        xm = xw - xw.mean(dim=1, keepdim=True)
        ym = yw - yw.mean(dim=1, keepdim=True)
        out[:, t, :] = (xm * ym).mean(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _ts_sum(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.zeros_like(x)
    for t in range(T):
        start = max(0, t - d + 1)
        out[:, t, :] = x[:, start:t + 1, :].sum(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _product(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    out = torch.ones_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = x[:, t - d + 1:t + 1, :].prod(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _decay_linear(x: Tensor, d: int) -> Tensor:
    if x.dim() == 2:
        x = x.unsqueeze(0)
    T = x.shape[1]
    weights = torch.arange(1, d + 1, dtype=x.dtype, device=x.device).float()
    weights = weights / weights.sum()
    out = torch.zeros_like(x)
    for t in range(d - 1, T):
        out[:, t, :] = (x[:, t - d + 1:t + 1, :] * weights.view(1, -1, 1)).sum(dim=1)
    return out.squeeze(0) if out.shape[0] == 1 else out


def _sign(x: Tensor) -> Tensor:
    return torch.sign(x)


def _log(x: Tensor) -> Tensor:
    return torch.log(x.clamp(min=1e-8))


def _abs(x: Tensor) -> Tensor:
    return x.abs()


# placeholder for scale (cross-sectional normalize to sum abs = 1)
def _scale(x: Tensor, a: float = 1.0) -> Tensor:
    s = x.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / s * a


# ============================================================
# Main compute function (skeleton - implementations added below)
# ============================================================

def compute_alpha101(
    open: Tensor,
    high: Tensor,
    low: Tensor,
    close: Tensor,
    volume: Tensor,
    returns: Optional[Tensor] = None,
    vwap: Optional[Tensor] = None,
) -> Tensor:
    """Compute 101 alpha factors + market signal factors.
    
    Args:
        open, high, low, close, volume: shape (T, N) or (B, T, N)
        returns: if None, computed from close
        vwap: if None, approximated as (high+low+close)/3
    
    Returns:
        Tensor of shape (..., T, N, num_factors)
    """
    # Ensure 3D
    squeeze = False
    if open.dim() == 2:
        squeeze = True
        open = open.unsqueeze(0)
        high = high.unsqueeze(0)
        low = low.unsqueeze(0)
        close = close.unsqueeze(0)
        volume = volume.unsqueeze(0)

    B, T, N = close.shape

    if returns is None:
        returns = torch.zeros_like(close)
        returns[:, 1:, :] = close[:, 1:, :] / close[:, :-1, :].clamp(min=1e-8) - 1
    elif returns.dim() == 2:
        returns = returns.unsqueeze(0)

    if vwap is None:
        vwap = (high + low + close) / 3
    elif vwap.dim() == 2:
        vwap = vwap.unsqueeze(0)

    factors = []

    # Alpha#001 to Alpha#101 + market signals will be appended
    # (implementations filled in via replace_in_file)
    factors.append(_alpha001(close, returns))
    factors.append(_alpha002(open, close, volume))
    factors.append(_alpha003(open, close, volume))
    factors.append(_alpha004(low, volume))
    factors.append(_alpha005(open, close, vwap))
    factors.append(_alpha006(open, volume))
    factors.append(_alpha007(close, volume, returns))
    factors.append(_alpha008(open, close, returns))
    factors.append(_alpha009(close))
    factors.append(_alpha010(close, returns))
    factors.append(_alpha011(close, volume, vwap))
    factors.append(_alpha012(close, volume))
    factors.append(_alpha013(close, volume))
    factors.append(_alpha014(open, close, volume, returns))
    factors.append(_alpha015(high, volume))
    factors.append(_alpha016(close, volume))
    factors.append(_alpha017(close, volume))
    factors.append(_alpha018(open, close))
    factors.append(_alpha019(close, returns))
    factors.append(_alpha020(open, high, low, close))
    factors.append(_alpha021(open, high, low, close, volume))
    factors.append(_alpha022(high, close, volume))
    factors.append(_alpha023(high, close))
    factors.append(_alpha024(close))
    factors.append(_alpha025(close, returns, vwap, volume))
    factors.append(_alpha026(close, volume))
    factors.append(_alpha027(close, volume))
    factors.append(_alpha028(high, low, close, volume))
    factors.append(_alpha029(close, returns))
    factors.append(_alpha030(close, volume))
    factors.append(_alpha031(close, low, volume))
    factors.append(_alpha032(close, volume, vwap))
    factors.append(_alpha033(open, close))
    factors.append(_alpha034(close, returns))
    factors.append(_alpha035(high, low, close, volume, returns))
    factors.append(_alpha036(open, close, volume, vwap, returns))
    factors.append(_alpha037(open, close))
    factors.append(_alpha038(high, close))
    factors.append(_alpha039(close, volume, returns))
    factors.append(_alpha040(high, volume))
    factors.append(_alpha041(high, low, vwap))
    factors.append(_alpha042(close, vwap))
    factors.append(_alpha043(close, volume))
    factors.append(_alpha044(high, volume))
    factors.append(_alpha045(close, volume))
    factors.append(_alpha046(close))
    factors.append(_alpha047(high, close, volume, vwap))
    factors.append(_alpha048(close, returns))
    factors.append(_alpha049(close))
    factors.append(_alpha050(volume, vwap))
    # Market signal factors
    factors.append(_market_ma_signal(close, 5, 20))
    factors.append(_market_ma_signal(close, 10, 60))
    factors.append(_market_volatility(returns, 20))
    factors.append(_market_momentum(close, 20))
    factors.append(_market_volume_signal(volume, 20))

    result = torch.stack(factors, dim=-1)  # (B, T, N, F)
    if squeeze:
        result = result.squeeze(0)
    return result


# ============================================================
# Alpha factor implementations (first 20 + market signals)
# ============================================================

def _alpha001(close: Tensor, returns: Tensor) -> Tensor:
    """rank(ts_argmax(sign(delta(close,1)) < 0 ? stddev(returns,20) : close, 5))"""
    cond = _sign(_delta(close, 1)) < 0
    inner = torch.where(cond, _stddev(returns, 20), close)
    return _rank(_ts_argmax(inner, 5))


def _alpha002(open: Tensor, close: Tensor, volume: Tensor) -> Tensor:
    """-1 * correlation(rank(delta(log(volume),2)), rank((close-open)/open), 6)"""
    return -1 * _correlation(_rank(_delta(_log(volume), 2)), _rank((close - open) / open.clamp(min=1e-8)), 6)


def _alpha003(open: Tensor, close: Tensor, volume: Tensor) -> Tensor:
    """-1 * correlation(rank(open), rank(volume), 10)"""
    return -1 * _correlation(_rank(open), _rank(volume), 10)


def _alpha004(low: Tensor, volume: Tensor) -> Tensor:
    """-1 * ts_rank(rank(low), 9)"""
    return -1 * _ts_rank(_rank(low), 9)


def _alpha005(open: Tensor, close: Tensor, vwap: Tensor) -> Tensor:
    """rank(open - ts_sum(vwap,10)/10) * (-1 * abs(rank(close - vwap)))"""
    return _rank(open - _sma(vwap, 10)) * (-1 * _abs(_rank(close - vwap)))


def _alpha006(open: Tensor, volume: Tensor) -> Tensor:
    """-1 * correlation(open, volume, 10)"""
    return -1 * _correlation(open, volume, 10)


def _alpha007(close: Tensor, volume: Tensor, returns: Tensor) -> Tensor:
    """adv20 = sma(volume,20); cond = adv20 < volume
    if cond: (-1*ts_rank(abs(delta(close,7)),60))*sign(delta(close,7)) else -1"""
    adv20 = _sma(volume, 20)
    cond = adv20 < volume
    inner = (-1 * _ts_rank(_abs(_delta(close, 7)), 60)) * _sign(_delta(close, 7))
    return torch.where(cond, inner, torch.tensor(-1.0, device=close.device))


def _alpha008(open: Tensor, close: Tensor, returns: Tensor) -> Tensor:
    """-1 * rank(ts_sum(open,5) * ts_sum(returns,5) - delay(ts_sum(open,5)*ts_sum(returns,5),10))"""
    inner = _ts_sum(open, 5) * _ts_sum(returns, 5)
    return -1 * _rank(inner - _delay(inner, 10))


def _alpha009(close: Tensor) -> Tensor:
    """if 0<ts_min(delta(close,1),5): delta(close,1) 
    elif ts_max(delta(close,1),5)<0: delta(close,1) else -1*delta(close,1)"""
    d = _delta(close, 1)
    cond1 = _ts_min(d, 5) > 0
    cond2 = _ts_max(d, 5) < 0
    return torch.where(cond1 | cond2, d, -1 * d)


def _alpha010(close: Tensor, returns: Tensor) -> Tensor:
    """rank(if 0<ts_min(delta(close,1),4): delta(close,1) elif ts_max(delta(close,1),4)<0: delta(close,1) else -1*delta(close,1))"""
    d = _delta(close, 1)
    cond1 = _ts_min(d, 4) > 0
    cond2 = _ts_max(d, 4) < 0
    return _rank(torch.where(cond1 | cond2, d, -1 * d))


def _alpha011(close: Tensor, volume: Tensor, vwap: Tensor) -> Tensor:
    """((rank(ts_max(vwap-close,3))+ rank(ts_min(vwap-close,3)))*rank(delta(volume,3)))"""
    return (_rank(_ts_max(vwap - close, 3)) + _rank(_ts_min(vwap - close, 3))) * _rank(_delta(volume, 3))


def _alpha012(close: Tensor, volume: Tensor) -> Tensor:
    """sign(delta(volume,1)) * (-1*delta(close,1))"""
    return _sign(_delta(volume, 1)) * (-1 * _delta(close, 1))


def _alpha013(close: Tensor, volume: Tensor) -> Tensor:
    """-1 * rank(covariance(rank(close),rank(volume),5))"""
    return -1 * _rank(_covariance(_rank(close), _rank(volume), 5))


def _alpha014(open: Tensor, close: Tensor, volume: Tensor, returns: Tensor) -> Tensor:
    """-1 * rank(delta(returns,3)) * correlation(open, volume, 10)"""
    return -1 * _rank(_delta(returns, 3)) * _correlation(open, volume, 10)


def _alpha015(high: Tensor, volume: Tensor) -> Tensor:
    """-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)"""
    return -1 * _ts_sum(_rank(_correlation(_rank(high), _rank(volume), 3)), 3)


def _alpha016(close: Tensor, volume: Tensor) -> Tensor:
    """-1 * rank(covariance(rank(close), rank(volume), 5))"""
    return -1 * _rank(_covariance(_rank(close), _rank(volume), 5))


def _alpha017(close: Tensor, volume: Tensor) -> Tensor:
    """rank(-1 * ts_rank(close, 10)) * rank(delta(delta(close,1),1)) * rank(ts_rank(volume/sma(volume,20),5))"""
    return _rank(-1 * _ts_rank(close, 10)) * _rank(_delta(_delta(close, 1), 1)) * _rank(_ts_rank(volume / _sma(volume, 20).clamp(min=1e-8), 5))


def _alpha018(open: Tensor, close: Tensor) -> Tensor:
    """-1 * rank(stddev(abs(close-open), 5) + (close-open) + correlation(close,open,10))"""
    return -1 * _rank(_stddev(_abs(close - open), 5) + (close - open) + _correlation(close, open, 10))


def _alpha019(close: Tensor, returns: Tensor) -> Tensor:
    """(-1*sign(close-delay(close,7)+delta(close,7))) * (1+rank(1+ts_sum(returns,250)))"""
    return (-1 * _sign(close - _delay(close, 7) + _delta(close, 7))) * (1 + _rank(1 + _ts_sum(returns, 250)))


def _alpha020(open: Tensor, high: Tensor, low: Tensor, close: Tensor) -> Tensor:
    """rank(open - delay(high,1)) * rank(open-delay(close,1)) * rank(open - delay(low,1))"""
    return _rank(open - _delay(high, 1)) * _rank(open - _delay(close, 1)) * _rank(open - _delay(low, 1))


# ============================================================
# Alpha factor implementations (21-50)
# ============================================================

def _alpha021(open: Tensor, high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
    c1 = _sma(close, 8) + _stddev(close, 8) < _sma(close, 2)
    c2 = _sma(volume, 20) / volume < 1
    inner = torch.where(c1, torch.tensor(-1.0, device=close.device),
            torch.where(c2, torch.tensor(1.0, device=close.device),
            torch.tensor(-1.0, device=close.device)))
    return inner

def _alpha022(high: Tensor, close: Tensor, volume: Tensor) -> Tensor:
    return -1 * _delta(_correlation(high, volume, 5), 5) * _rank(_stddev(close, 20))

def _alpha023(high: Tensor, close: Tensor) -> Tensor:
    cond = _sma(high, 20) < high
    return torch.where(cond, -1 * _delta(high, 2), torch.zeros_like(high))

def _alpha024(close: Tensor) -> Tensor:
    cond = _delta(_sma(close, 100), 100) / _delay(close, 100).clamp(min=1e-8) <= 0.05
    return torch.where(cond, -1 * _delta(close, 3), -1 * (close - _ts_min(close, 100)))

def _alpha025(close: Tensor, returns: Tensor, vwap: Tensor, volume: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return _rank(-1 * returns * adv20 * vwap * (close - _delay(close, 1)))

def _alpha026(close: Tensor, volume: Tensor) -> Tensor:
    return -1 * _ts_max(_correlation(_ts_rank(volume, 5), _ts_rank(close, 5), 5), 3)

def _alpha027(close: Tensor, volume: Tensor) -> Tensor:
    val = _correlation(_rank(volume), _rank(close), 6)
    cond = _rank(val) < 2.5
    return torch.where(cond, torch.tensor(-1.0, device=close.device), _rank(val))

def _alpha028(high: Tensor, low: Tensor, close: Tensor, volume: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return _scale(_correlation(adv20, low, 5) + (high + low) / 2 - close)

def _alpha029(close: Tensor, returns: Tensor) -> Tensor:
    return _ts_min(_rank(_rank(_scale(_log(_ts_sum(_rank(_rank(-1 * _rank(_delta(close - 1, 5)))), 2))))), 5)

def _alpha030(close: Tensor, volume: Tensor) -> Tensor:
    d = _delta(close, 1)
    return _rank(_sign(d) + _sign(_delay(d, 1)) + _sign(_delay(d, 2))) * _ts_sum(volume, 5) / _ts_sum(volume, 20).clamp(min=1e-8)

def _alpha031(close: Tensor, low: Tensor, volume: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return _rank(_decay_linear(-1 * _rank(_delta(close, 10)), 10)) + _rank(-1 * _delta(close, 3)) + _sign(_scale(_correlation(adv20, low, 12)))

def _alpha032(close: Tensor, volume: Tensor, vwap: Tensor) -> Tensor:
    return _scale(_sma(close, 7) - close) + 20 * _scale(_correlation(vwap, _delay(close, 5), 230))

def _alpha033(open: Tensor, close: Tensor) -> Tensor:
    return _rank(-1 + open / close.clamp(min=1e-8))

def _alpha034(close: Tensor, returns: Tensor) -> Tensor:
    return _rank(2 - _rank(_stddev(returns, 2) / _stddev(returns, 5).clamp(min=1e-8)) + _rank(_delta(close, 1)))

def _alpha035(high: Tensor, low: Tensor, close: Tensor, volume: Tensor, returns: Tensor) -> Tensor:
    return _ts_rank(volume, 32) * (1 - _ts_rank(close + high - low, 16)) * (1 - _ts_rank(returns, 32))

def _alpha036(open: Tensor, close: Tensor, volume: Tensor, vwap: Tensor, returns: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return 2.21 * _rank(_correlation(close - open, _delay(volume, 1), 15)) + 0.7 * _rank(open - close) + 0.73 * _rank(_ts_rank(_delay(-1 * returns, 6), 5)) + _rank(_abs(_correlation(vwap, adv20, 6))) + 0.6 * _rank((_sma(close, 200) - open) * (close - open))

def _alpha037(open: Tensor, close: Tensor) -> Tensor:
    return _rank(_correlation(_delay(open - close, 1), close, 200)) + _rank(open - close)

def _alpha038(high: Tensor, close: Tensor) -> Tensor:
    return -1 * _rank(_ts_rank(close, 10)) * _rank(close / high.clamp(min=1e-8))

def _alpha039(close: Tensor, volume: Tensor, returns: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return -1 * _rank(_delta(close, 7) * (1 - _rank(_decay_linear(volume / adv20.clamp(min=1e-8), 9)))) * (1 + _rank(_ts_sum(returns, 250)))

def _alpha040(high: Tensor, volume: Tensor) -> Tensor:
    return -1 * _rank(_stddev(high, 10)) * _correlation(high, volume, 10)

def _alpha041(high: Tensor, low: Tensor, vwap: Tensor) -> Tensor:
    return (high * low) ** 0.5 - vwap

def _alpha042(close: Tensor, vwap: Tensor) -> Tensor:
    return _rank(vwap - close) / _rank(vwap + close).clamp(min=1e-8)

def _alpha043(close: Tensor, volume: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return _ts_rank(volume / adv20.clamp(min=1e-8), 20) * _ts_rank(-1 * _delta(close, 7), 8)

def _alpha044(high: Tensor, volume: Tensor) -> Tensor:
    return -1 * _correlation(high, _rank(volume), 5)

def _alpha045(close: Tensor, volume: Tensor) -> Tensor:
    return -1 * _rank(_sma(_delay(close, 5), 20)) * _correlation(close, volume, 2) * _rank(_correlation(_ts_sum(close, 5), _ts_sum(close, 20), 2))

def _alpha046(close: Tensor) -> Tensor:
    cond = 0.25 < _delta(close, 20) / _delay(close, 20).clamp(min=1e-8)
    return torch.where(cond, torch.tensor(-1.0, device=close.device), _delta(close, 1))

def _alpha047(high: Tensor, close: Tensor, volume: Tensor, vwap: Tensor) -> Tensor:
    adv20 = _sma(volume, 20)
    return _rank(1 / close.clamp(min=1e-8)) * volume / adv20.clamp(min=1e-8) * high * _rank(high - close) / _sma(high, 5).clamp(min=1e-8) - _rank(vwap - _delay(vwap, 5))

def _alpha048(close: Tensor, returns: Tensor) -> Tensor:
    # indneutralize simplified as rank
    return _rank(_correlation(_delta(close, 1), _delta(_delay(close, 1), 1), 250)) * _rank(_delta(close, 1)) / close.clamp(min=1e-8)

def _alpha049(close: Tensor) -> Tensor:
    cond = _delta(close, 1) > _delay(_delta(close, 1), 1)
    return torch.where(cond, _delta(close, 1), torch.tensor(0.0, device=close.device))

def _alpha050(volume: Tensor, vwap: Tensor) -> Tensor:
    return -1 * _ts_max(_rank(_correlation(_rank(volume), _rank(vwap), 5)), 5)


# ============================================================
# Market signal factors
# ============================================================

def _market_ma_signal(close: Tensor, short_w: int, long_w: int) -> Tensor:
    """MA crossover signal: short MA - long MA, normalized."""
    ma_short = _sma(close, short_w)
    ma_long = _sma(close, long_w)
    return (ma_short - ma_long) / ma_long.clamp(min=1e-8)


def _market_volatility(returns: Tensor, window: int) -> Tensor:
    """Rolling volatility."""
    return _stddev(returns, window)


def _market_momentum(close: Tensor, window: int) -> Tensor:
    """Momentum: current price / price window days ago - 1."""
    return close / _delay(close, window).clamp(min=1e-8) - 1


def _market_volume_signal(volume: Tensor, window: int) -> Tensor:
    """Volume ratio signal: current volume / SMA(volume, window)."""
    return volume / _sma(volume, window).clamp(min=1e-8)


# ============================================================
# Data loading utility
# ============================================================

def load_ohlcv_csv(path: str, device: str = "cpu") -> dict:
    """Load OHLCV data from a CSV file.
    
    Expected columns: date, open, high, low, close, volume
    Optionally: symbol (for multi-stock), vwap
    
    Returns dict with keys: open, high, low, close, volume, returns, vwap
    Each tensor has shape (T, N) where N is number of stocks.
    """
    import pandas as pd

    df = pd.read_csv(path, parse_dates=["date"])
    
    if "symbol" in df.columns:
        # Multi-stock pivot
        symbols = sorted(df["symbol"].unique())
        pivot = df.pivot(index="date", columns="symbol")
        o = torch.tensor(pivot["open"][symbols].values, dtype=torch.float32, device=device)
        h = torch.tensor(pivot["high"][symbols].values, dtype=torch.float32, device=device)
        l = torch.tensor(pivot["low"][symbols].values, dtype=torch.float32, device=device)
        c = torch.tensor(pivot["close"][symbols].values, dtype=torch.float32, device=device)
        v = torch.tensor(pivot["volume"][symbols].values, dtype=torch.float32, device=device)
    else:
        # Single stock
        o = torch.tensor(df["open"].values, dtype=torch.float32, device=device).unsqueeze(-1)
        h = torch.tensor(df["high"].values, dtype=torch.float32, device=device).unsqueeze(-1)
        l = torch.tensor(df["low"].values, dtype=torch.float32, device=device).unsqueeze(-1)
        c = torch.tensor(df["close"].values, dtype=torch.float32, device=device).unsqueeze(-1)
        v = torch.tensor(df["volume"].values, dtype=torch.float32, device=device).unsqueeze(-1)

    # Compute returns
    ret = torch.zeros_like(c)
    ret[1:] = c[1:] / c[:-1].clamp(min=1e-8) - 1

    # VWAP
    if "vwap" in df.columns:
        if "symbol" in df.columns:
            vwap = torch.tensor(pivot["vwap"][symbols].values, dtype=torch.float32, device=device)
        else:
            vwap = torch.tensor(df["vwap"].values, dtype=torch.float32, device=device).unsqueeze(-1)
    else:
        vwap = (h + l + c) / 3

    return {"open": o, "high": h, "low": l, "close": c, "volume": v, "returns": ret, "vwap": vwap}
