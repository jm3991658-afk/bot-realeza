import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Literal, Optional

Signal = Literal["CALL", "PUT", "NONE"]


@dataclass(frozen=True)
class BotConfig:
    rsi_period: int = 14
    adx_period: int = 25
    adx_min: float = 20.0
    rsi_call: float = 52.0
    rsi_put: float = 48.0
    expiration_minutes: int = 30
    use_g1: bool = False


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    result = result.where(avg_loss != 0, 100.0)
    return result.where(avg_gain != 0, 0.0)


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 25):
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    previous_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - previous_close).abs(),
        (low - previous_close).abs(),
    ], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_smoothed = plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    minus_smoothed = minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    plus_di = 100 * plus_smoothed / atr.replace(0, np.nan)
    minus_di = 100 * minus_smoothed / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_line, plus_di, minus_di


def validate_candles(df: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("O índice precisa ser DatetimeIndex.")
    return df.sort_index().astype(float).dropna(subset=list(required))


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()


def add_indicators(df: pd.DataFrame, cfg: BotConfig) -> pd.DataFrame:
    out = df.copy()
    out["rsi"] = rsi(out["close"], cfg.rsi_period)
    out["adx"], out["plus_di"], out["minus_di"] = adx(
        out["high"], out["low"], out["close"], cfg.adx_period
    )
    return out


def generate_signals(candles: pd.DataFrame, cfg: BotConfig = BotConfig()) -> pd.DataFrame:
    """Recebe candles de 1 minuto e produz sinais no fechamento das velas M15."""
    base = validate_candles(candles)
    m15 = add_indicators(resample_ohlc(base, "15min"), cfg)
    m30 = add_indicators(resample_ohlc(base, "30min"), cfg)

    # Só utiliza a última vela M30 já encerrada no instante de cada fechamento M15.
    trend = m30[["adx", "plus_di", "minus_di"]].rename(columns={
        "adx": "adx_m30", "plus_di": "plus_di_m30", "minus_di": "minus_di_m30"
    }).reindex(m15.index, method="ffill")
    out = m15.join(trend)

    call = (
        (out["adx_m30"] >= cfg.adx_min)
        & (out["plus_di_m30"] > out["minus_di_m30"])
        & (out["adx"] >= cfg.adx_min)
        & (out["plus_di"] > out["minus_di"])
        & (out["rsi"] >= cfg.rsi_call)
        & (out["close"] > out["open"])
    )
    put = (
        (out["adx_m30"] >= cfg.adx_min)
        & (out["minus_di_m30"] > out["plus_di_m30"])
        & (out["adx"] >= cfg.adx_min)
        & (out["minus_di"] > out["plus_di"])
        & (out["rsi"] <= cfg.rsi_put)
        & (out["close"] < out["open"])
    )

    out["signal"] = np.select([call, put], ["CALL", "PUT"], default="NONE")
    out["expiration"] = out.index + pd.Timedelta(minutes=cfg.expiration_minutes)
    return out


def latest_signal(candles: pd.DataFrame, cfg: BotConfig = BotConfig()) -> Optional[dict]:
    signals = generate_signals(candles, cfg)
    actionable = signals[signals["signal"] != "NONE"]
    if actionable.empty:
        return None
    time = actionable.index[-1]
    row = actionable.iloc[-1]
    return {
        "time": time.isoformat(),
        "signal": row["signal"],
        "expiration": row["expiration"].isoformat(),
        "rsi_m15": round(float(row["rsi"]), 2),
        "adx_m15": round(float(row["adx"]), 2),
        "adx_m30": round(float(row["adx_m30"]), 2),
    }


if __name__ == "__main__":
    # CSV: time,open,high,low,close; idealmente candles de 1 minuto.
    candles = pd.read_csv("candles.csv", parse_dates=["time"]).set_index("time")
    result = latest_signal(candles)
    print(result or "Nenhum sinal confirmado.")
