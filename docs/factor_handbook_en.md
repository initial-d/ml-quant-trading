# Factor Handbook

This handbook describes in detail the design ideas, motivations, and principles behind all **204 handcrafted quantitative factors** in `mlquant.features.legacy_factors`.
Together with the **9 selected Alpha101 factors** in `mlquant.features.alpha101`, the system provides a total of **213 feature dimensions**.
Users can select an appropriate factor subset according to their strategy requirements.

> **Research documentation only.** The interpretations below are hypotheses and
> implementation notes, not claims of predictive power or investment advice.
> Validate every factor out of sample with realistic costs before drawing conclusions.
> If the prose and implementation differ, the registered factor code is authoritative.
>
> [Chinese handbook / 中文版](factor_handbook.md)

---

## Usage

```python
from mlquant.features import compute_legacy_set, LEGACY_REGISTRY

# Compute the unified 213-factor set (204 handcrafted + 9 selected Alpha101)
factors, mask, names = compute_legacy_set(panel)

# Select a specific factor subset
factors, mask, names = compute_legacy_set(
    panel, names=("best_001", "old_027", "better_003")
)

# View all available factor names
print(list(LEGACY_REGISTRY.keys()))
```

---

## Factor Overview

| Family | Count | Core Idea | Module |
|--------|-------|-----------|--------|
| [better_*](#better-family-28-factors) | 28 | VWAP deviation + volume-weighted momentum | `_factors_better.py` |
| [best_*](#best-family-21-factors) | 21 | Intraday close-location momentum | `_factors_best.py` |
| [old_*](#old-family-50-factors) | 50 | Classic Alpha signals | `_factors_old.py` |
| [stock_*](#stock-family-22-factors) | 22 | Stock-specific derived series | `_factors_stock.py` |
| [extra_*](#extra-family-14-factors) | 14 | Turnover and trading-value features | `_factors_extra.py` |
| [add_*](#add-family-30-factors) | 30 | Supplementary factor variants | `_factors_add.py` |
| [change_*](#change-family-5-factors) | 5 | Short-window changes in velocity | `_factors_change.py` |
| [original_*](#original-family-28-factors) | 28 | Basic price-volume statistics | `_factors_original.py` |
| [cs_rank_*](#cs_rank-family-6-factors) | 6 | Market-breadth signals | `_factors_market.py` |

---

## better Family (28 Factors)

VWAP (volume-weighted average price) is a core benchmark price for institutional trading. This family builds factors around deviations between VWAP and the closing price,
abnormal volume, intraday volatility, and related dimensions to capture institutional capital flows and short-term pricing dislocations.

| Factor | Idea and Rationale |
|--------|--------------------|
| `better_001` | **Overnight gap × log volume**. The overnight gap (`open/prev_close - 1`) reflects information shocks; multiplying it by `log₂(volume)` amplifies the signal when volume is high. A high-volume gap is more likely to be driven by genuine information rather than noise. |
| `better_002` | **Overnight gap × relative volume**. Similar to `better_001`, but uses the ratio of current volume to its 5-day average as the weight. Gap signals are more credible when volume expands. |
| `better_003` | **Gap × VWAP deviation × intraday range**. A three-way interaction among the direction of the overnight gap, the deviation between VWAP and the close (institutional buying/selling pressure), and the intraday range (volatility). It captures the direction of institutional flows in high-volatility environments. |
| `better_004` | **1-day change in VWAP location**. The change in VWAP's location within the intraday high-low range reflects short-term shifts in the center of institutional trading. The negative sign makes a downward shift in VWAP location (increased selling pressure) a positive signal. |
| `better_005` | **5-day change in VWAP location**. Uses the same logic as `better_004` over a longer window to capture medium-term trends in institutional behavior. |
| `better_006` | **5-day mean VWAP location**. The average location of VWAP within the intraday range. Persistently high values indicate that institutions continue to transact near the upper end (buying pressure); low values indicate the opposite. |
| `better_007` | **Intraday range × volume change**. `(high-low)/close` measures volatility and is multiplied by the change in volume. Expanding volatility accompanied by rising volume often suggests trend continuation. |
| `better_008` | **Intraday range / log volume**. Volatility divided by log volume measures "volatility per unit of liquidity," a variant of Amihud illiquidity. A high value implies poor liquidity and large price impact. |
| `better_009` | **Conditional downside accumulation (20 days)**. Accumulates `\|VWAP rate of change\|/log(amount)` only on days when VWAP declines, measuring the "pain" of a selloff. A high value indicates that recent declines occurred with low liquidity and may imply oversold conditions. |
| `better_010` | **Conditional downside accumulation (10 days)**. Uses the same logic as `better_009` over a shorter window, making it more sensitive to recent declines. |
| `better_011` | **Multi-horizon support and resistance**. Extends the current day's range with the previous day's high and low, calculates the close's location within that expanded range, and averages sums over 6/12/24-day windows. This combines support and resistance strength across multiple horizons. |
| `better_012` | **VWAP-based support and resistance**. Similar to `better_011`, but uses VWAP instead of the close as the reference point, better reflecting institutions' actual transaction costs. |
| `better_013` | **VWAP deviation × log trading value**. `(vwap/close - 1)` measures the premium or discount of the institutional transaction price relative to the close; multiplying it by log trading value amplifies large-trade signals. |
| `better_014` | **4-day mean VWAP deviation**. A smoothed version of `better_013` that reduces single-day noise and captures persistent institutional premiums or discounts. |
| `better_015` | **VWAP-deviation acceleration (EMA-smoothed)**. First calculates VWAP's deviation from its 10-day mean, then takes a 5-day difference (acceleration), and finally smooths it with `EMA(1/20)`. It captures turning points in the VWAP-deviation trend. |
| `better_016` | **VWAP-deviation acceleration (short window)**. Uses the same logic as `better_015` with shorter parameters (6-day mean, 3-day difference, `EMA 1/12`), making it more sensitive to short-term changes. |
| `better_017` | **Trading-value deviation acceleration (short window)**. Takes the 3-day difference of trading value's deviation from its 6-day mean and applies EMA smoothing. It captures accelerating capital inflows or outflows. |
| `better_018` | **Trading-value deviation acceleration (long window)**. Similar to `better_017`, but uses a 10-day mean and a 5-day difference to capture more persistent changes in capital flow. |
| `better_019` | **Triple-EMA volume rate of change**. Applies EMA smoothing three times to `log(volume)` and then calculates its rate of change, similar to the TRIX indicator. Triple smoothing filters noise effectively and captures trending changes in volume. |
| `better_020` | **Coefficient of variation of Amihud illiquidity**. The 10-day standard deviation of `\|return\|/volume` divided by its mean (CV), measuring liquidity stability. A high CV indicates unstable liquidity and substantial price-impact risk. |
| `better_021` | **VWAP/previous-close ratio**. Directly measures the premium of the day's institutional average transaction price relative to the previous close. A value greater than 1 indicates that institutions traded above the previous close (buying dominated). |
| `better_022` | **VWAP premium × log trading value**. `(vwap/prev_close - 1) × log₂(amount)` weights the VWAP-premium signal by trading value; the premium associated with large trades is more meaningful. |
| `better_023` | **4-day mean VWAP premium**. A smoothed version of `better_022` that reduces day-to-day volatility noise. |
| `better_024` | **VWAP-close spread extremes × volume change**. The sum of the 5-day maximum and minimum of `(vwap-close)`, multiplied by the 5-day volume change. The sum of the extremes reflects the direction of the spread distribution, while the volume change provides confirmation. |
| `better_025` | **VWAP-close spread extremes × short-term volume change**. Similar to `better_024`, but uses a shorter 3-day window and is more sensitive to recent extremes. |
| `better_026` | **Volatility expansion ratio**. Short-term EMA of the intraday range divided by long-term EMA of the intraday range, a dynamic analogue of Bollinger Band width. A ratio greater than 1 indicates expanding volatility and may foreshadow the start of a trend. |
| `better_027` | **Conditional downside accumulation (VWAP vs. previous close)**. Accumulates the signal only on days when `VWAP < previous close`, measuring how frequently and strongly institutions transact below the previous close. |
| `better_028` | **Conditional downside accumulation (VWAP vs. current close)**. Accumulates the signal only on days when `VWAP < close`, indicating that earlier intraday transactions occurred at lower prices followed by a late-session rally, potentially reflecting retail performance chasing. |

---

## best Family (21 Factors)

This family builds factors around "close location." Close location is defined as the relative position of the closing price within the intraday high-low
range: `(2C - H - L) / (H - L)`. A value near +1 means the close is near the high (bulls in control),
while a value near -1 means the close is near the low (bears in control).

| Factor | Idea and Rationale |
|--------|--------------------|
| `best_001` | **5-day time-series rank of close location**. Where does today's close location rank over the last 5 days? A high rank indicates strengthening bullish pressure. |
| `best_002` | **10-day time-series rank of close location**. A longer-window version that captures medium-term trends. |
| `best_003` | **20-day time-series rank of close location**. A monthly-horizon trend in close location. |
| `best_004` | **3-day change in close location**. The change in close location over 3 days; a positive value indicates that bullish pressure has strengthened over the short term. |
| `best_005` | **5-day change in close location**. Similar to `best_004` but over a longer window. |
| `best_006` | **EMA of close location**. Smooths close location with an exponential moving average (`α=1/10`), filtering day-to-day noise and extracting the trend. |
| `best_007` | **Cross-sectional rank of close location**. The stock's relative close-location rank among all stocks that day. A high rank indicates stronger bullish control relative to the market. |
| `best_008` | **3-day mean VWAP/close ratio**. VWAP persistently above the close indicates substantial intraday selling pressure (institutions distributing at higher prices); the reverse indicates buying dominance. |
| `best_009` | **`EMA(VWAP) / EMA(close)`**. The ratio of EMA-smoothed VWAP to EMA-smoothed close captures persistent displacement of the institutional trading center relative to the close. |
| `best_010` | **`(VWAP/close - 1) × volume`**. VWAP premium multiplied by volume measures the "total volume transacted at an institutional premium." |
| `best_011` | **Cross-sectional rank of the overnight gap**. The cross-sectional rank of `open/prev_close - 1` measures the stock's overnight information shock relative to the market. |
| `best_012` | **5-day mean overnight gap**. A smoothed gap signal; persistent positive gaps indicate a continuing inflow of favorable information. |
| `best_013` | **EMA of the overnight gap**. Smooths the gap signal with an EMA, assigning greater weight to recent gaps. |
| `best_014` | **Correlation between volume rank and range rank**. The 5-day correlation between the time-series rank of volume and the time-series rank of intraday range. A high correlation means that "high volume accompanies high volatility," potentially indicating trend continuation. |
| `best_015` | **Cross-sectional rank of intraday range**. The cross-sectional rank of `(high-low)/close` measures the stock's volatility activity relative to the market that day. |
| `best_016` | **Range/√volume**. Volatility divided by the square root of volume measures price movement per unit of trading volume, an approximation of Kyle's lambda. |
| `best_017` | **1-day change in `EMA(close)` (`α=1/5`)**. The daily change in a short-term MA, similar to the slope of MACD's fast line. |
| `best_018` | **1-day change in `EMA(close)` (`α=1/10`)**. The slope of a medium-term MA. |
| `best_019` | **1-day change in `EMA(close)` (`α=1/20`)**. The slope of a long-term MA, capturing slow-moving trends. |
| `best_020` | **EMA of capital flow**. The EMA of `(close - vwap) × volume`; a positive value indicates that the close is above VWAP with high volume, implying net capital inflow. |
| `best_021` | **VWAP-spread extremes × volume change**. The 3-day maximum plus minimum of `(vwap-close)×vol`, multiplied by the volume change, captures volume confirmation following extreme deviations. |

---
## old Family (50 Factors)

Classic quantitative Alpha signals derived from WorldQuant Alpha101 and related variants.
They cover momentum, reversal, liquidity, volatility, and other dimensions commonly
discussed in quantitative research.

| Factor | Idea and Rationale |
|--------|--------------------|
| `old_027` | **Volume rank × VWAP-deviation rank**. Stocks with high volume and VWAP above the close (institutions distributing at higher prices) receive high scores. |
| `old_028` | **10-day correlation between open rank and volume rank**. Do stocks with higher opening prices also have higher volume? A positive correlation indicates that the market is chasing high-priced stocks. |
| `old_029` | **12-day minimum of the ranked close-volume correlation**. An extremely low price-volume-correlation rank captures extreme price-volume divergence. |
| `old_030` | **Direction of price change × (1 - volume-change rank)**. Stocks whose prices rise on low-ranked volume (a low-volume advance) receive high scores, potentially reflecting supply contraction. |
| `old_031` | **Price-change rank × volume-rate-of-change rank**. Captures breakout signals in stocks with simultaneous large changes in price and volume. |
| `old_032` | **MA deviation + correlation between VWAP and delayed close**. Combines short-term mean-reversion pressure with a long-term price-memory effect. |
| `old_033` | **Rank of the open/close ratio**. The rank of `open/close - 1` measures intraday direction. An open above the close (a bearish candle) ranks highly. |
| `old_034` | **Volatility ratio + price-change rank**. Adds the rank of short-term/long-term volatility to the price-change rank, capturing volatility clustering. |
| `old_035` | **Volume time-series rank × (1 - price-plus-range time-series rank)**. Stocks with a high volume rank but low price rank, a reversal signal after a high-volume decline. |
| `old_036` | **Ranked correlation between delayed (open-close) and close + open-close rank**. Measures the association between yesterday's intraday direction and today's close. |
| `old_037` | **`-rank(open-delay(high)) × rank(open-delay(close)) × rank(open-delay(low))`**. A three-way deviation of the open from the previous day's high, close, and low, capturing mean-reversion pressure after a gap. |
| `old_038` | **`-(rank(open) ^ rank(close/vwap))`**. A power transformation of the opening-price rank that nonlinearly captures the price-VWAP relationship. |
| `old_039` | **`-rank(Δclose,7) × (1 - rank(EMA(volume×ret)))`**. Medium-term momentum reversal multiplied by the EMA rank of volume-weighted returns. |
| `old_040` | **`-rank(std(high,10)) × corr(high,volume,10)`**. Stocks with volatile highs and a positive high-volume correlation may be speculatively overheated. |
| `old_041` | **`(high×low)^0.5 - vwap`**. The difference between the geometric mean price and VWAP measures skewness in the intraday price distribution. |
| `old_042` | **`rank(vwap-close) / rank(vwap+close)`**. The relative strength of VWAP deviation, representing standardized institutional buying/selling pressure. |
| `old_043` | **Relative-volume time-series rank × negative price-change time-series rank**. Stocks with expanding relative volume and falling prices may be oversold after panic selling. |
| `old_044` | **`-corr(high, rank(volume), 5)`**. A negative correlation between the high and volume rank: high prices without high volume may indicate a false breakout. |
| `old_045` | **`-rank(mean delayed close) × corr(close, volume, 2)`**. An interaction between the historical-price-level rank and short-term price-volume correlation. |
| `old_046` | **Multi-horizon MA deviation**. `(MA3+MA6+MA12+MA24)/(4×close) - 1` combines MA support/resistance across multiple horizons. |
| `old_047` | **Williams %R (6 days)**. `(highest high-close)/(highest high-lowest low)×100`, a classic overbought/oversold indicator. |
| `old_048` | **`-Δclose × volume / EMA(volume)`**. A signal for price declines accompanied by relatively high volume, similar to an OBV variant. |
| `old_049` | **20-day cumulative downside**. Sums losses only on down days, measuring recent "total pain." |
| `old_050` | **`-max(rank(corr(rank(vol), rank(vwap), 5)), 5)`**. An extreme value of the correlation between volume rank and VWAP rank, capturing extreme states in the liquidity-price relationship. |
| `old_051` | **12-day cumulative downside**. Similar to `old_049` but over a shorter window, making it more sensitive to recent declines. |
| `old_052` | **Upward-strength/downward-strength ratio (26 days)**. An RSI-like construction: cumulative gains divided by cumulative losses. |
| `old_053` | **Fraction of up days over 12 days**. A simple momentum indicator measuring how many of the last 12 sessions closed higher. |
| `old_054` | **`-rank(variability of \|close-open\| + \|close-open\|) + rank(corr(close,open,10))`**. A hedge between intraday movement and close-open correlation. |
| `old_055` | **6-day correlation between price-location rank and volume rank**. Measures the association between the close's location within the 12-day high-low range and volume. |
| `old_056` | **`-rank(cumulative return) × rank(price-volume correlation) × rank(volatility)`**. A three-factor interaction among momentum, the price-volume relationship, and volatility. |
| `old_057` | **`close / EMA(close, 1/30) - 1`**. The close's deviation from its long-term EMA, a classic MA-deviation ratio. |
| `old_058` | **`-volume × Δclose`**. The negative of volume-weighted price change: a high-volume advance is a negative signal under reversal logic. |
| `old_059` | **Rank of `mean(volume × \|ret\|, 20)`**. The rank of the 20-day mean of "volume × absolute return" measures trading activity. |
| `old_060` | **Rank of open relative to the 12-day low - rank of squared cumulative return**. A hedge between price location and squared momentum. |
| `old_061` | **`rank(vwap - 16-day lowest vwap)`**. The rank of VWAP's distance from its recent low measures VWAP's upside movement. |
| `old_062` | **`rank(corr(vwap, cumulative average volume, 5))`**. The rank of the correlation between VWAP and the long-term volume trend. |
| `old_063` | **`max(rank(corr(rank(vwap), rank(volume), 4)), 8)`**. The 8-day extreme of the correlation between VWAP rank and volume rank. |
| `old_064` | **`rank(corr(weighted price, cumulative average volume, 11))`**. The correlation between price and the long-term volume trend. |
| `old_065` | **`rank(corr(weighted price, short-term cumulative average volume, 6))`**. Similar to `old_064` but over a shorter window. |
| `old_066` | **`rank(Δvwap, 4)`**. The cross-sectional rank of the 4-day change in VWAP, a simple VWAP-momentum signal. |
| `old_067` | **`(high - 6-day highest high) / 6-day highest high × rank(corr(vwap, average volume, 4))`**. Drawdown from the recent high multiplied by price-volume correlation. |
| `old_068` | **`rank(corr(high, average volume, 9)) × rank(corr(close, volume, 4))`**. An interaction between the high-average-volume correlation and the close-volume correlation. |
| `old_069` | **`sum(max(rank(corr(ts_rank(close), ts_rank(vol), 4)), 0), 3)`**. The cumulative strength of positive correlation between price rank and volume rank. |
| `old_070` | **`rank(Δvwap, 2)`**. The rank of the 2-day VWAP change, an ultra-short-term VWAP-momentum signal. |
| `old_071` | **`ts_rank(corr(ts_rank(close,3), ts_rank(vol,3), 18), 4)`**. The time-series rank of short-term price-volume rank correlation. |
| `old_072` | **`rank(EMA(corr(average volume,low,4))) + rank(EMA(corr(rank(vwap),rank(vol),4)))`**. The sum of EMA-smoothed ranks for the low-volume relationship and the VWAP-volume relationship. |
| `old_073` | **`-ts_rank(EMA(Δvwap,5), 3)`**. Takes the negative time-series rank of EMA-smoothed VWAP changes, a VWAP-momentum reversal signal. |
| `old_074` | **`rank(corr(close, long-term cumulative average volume, 15))`**. The correlation between the closing price and the long-term volume trend. |
| `old_075` | **`rank(corr(rank(vwap),rank(vol),5)) - rank(corr(rank(close),rank(vol),5))`**. The difference between the VWAP-volume relationship and the close-volume relationship captures intraday pricing dislocations. |
| `old_076` | **`rank(Δ(corr(vwap, volume, 4), 3))`**. The rank of the rate of change in VWAP-volume correlation captures structural changes in the price-volume relationship. |

---

## stock Family (22 Factors)

These factors are built around price-volume series derived from each stock itself and emphasize stock-specific microstructure characteristics, including
liquidity, momentum acceleration, and technical patterns.

| Factor | Idea and Rationale |
|--------|--------------------|
| `stock_001` | **4-day correlation between `Δlog(volume)` and intraday return**. The short-term correlation between the rate of change in volume and intraday price movement. A positive correlation means "up on high volume, down on low volume," a healthy trend signal. |
| `stock_002` | **6-day correlation between `Δlog(volume)` and intraday return**. The same as `stock_001` but over a longer, more stable window. |
| `stock_003` | **`rank(close - 15-day highest vwap) ^ Δclose`**. The rank of the close's distance from the VWAP high, modulated by price change as an exponent. This nonlinearly captures breakout/drawdown intensity. |
| `stock_004` | **EMA of acceleration in price deviation from its mean**. The 3-day rate of change in `(close - MA6) / MA6`, smoothed with `EMA(1/12)`. It captures accelerating turning points in mean reversion/deviation. |
| `stock_005` | **6-day return × (volume + 1)**. Medium-term momentum multiplied by volume; the momentum signal is stronger for high-volume advances. |
| `stock_006` | **`(close - MA12) / MA12`**. A classic 12-day MA-deviation ratio; positive values indicate overbought conditions and negative values indicate oversold conditions. |
| `stock_007` | **Delayed 5-day low - 5-day low × momentum rank × volume rank**. The rebound distance from the bottom, multiplied by medium-to-long-term momentum rank and volume rank. A bottom-reversal signal with three-way confirmation. |
| `stock_008` | **Rank of `-(5-day sum of open×return - 10-day delay)`**. The change in open-weighted short-term momentum relative to 10 days earlier, negated to express momentum reversal. |
| `stock_009` | **Composite multi-horizon MA ratio**. `(MA3+MA6+MA12+MA24) / (4×close)` combines MA support/resistance across multiple horizons. A value greater than 1 means price is below its MAs (oversold). |
| `stock_010` | **6-day volatility of `log(trading value)`**. The standard deviation of log trading value measures the stability of capital flow. High volatility indicates unstable inflows and outflows. |
| `stock_011` | **CCI variant**. `(typical price - MA12(typical price)) / (0.015 × MAD)`, a classic Commodity Channel Index that measures how far price deviates from its statistically normal range. |
| `stock_012` | **Volume-RSI variant**. `EMA(max(Δvol,0)) / EMA(\|Δvol\|) × 100` measures the relative strength of increases in volume. A high value indicates persistently expanding volume. |
| `stock_013` | **`Δ(vwap-close) / Δ(vwap+close)`**. The rate of change in VWAP deviation divided by the rate of change in `VWAP+close`, a standardized indicator of changes in institutional behavior. |
| `stock_014` | **`(close / delay(close,12) - 1) × volume`**. The 12-day momentum multiplied by volume, signaling a high-volume medium-term breakout. |
| `stock_015` | **Conditional downside liquidity (20 days)**. Accumulates `\|ret\|/log(amount)` only on down days, measuring liquidity stress during declines. A high value indicates scarce liquidity during selloffs (panic selling). |
| `stock_016` | **Multi-time-frame Williams %R composite**. Combines Williams %R over 6/12/24-day windows to summarize short-, medium-, and long-term overbought/oversold conditions. |
| `stock_017` | **`-ret × average volume × vwap × (high-close)`**. The negative four-way interaction among return, average volume, VWAP, and the upper shadow. A long upper shadow on an up day with high volume may signal distribution. |
| `stock_018` | **5-day return - 20-day return**. Short-term momentum minus medium-term momentum captures momentum acceleration/deceleration. A positive value means short-term momentum exceeds medium-term momentum, indicating trend acceleration. |
| `stock_019` | **`(low-close) × open^5 / ((close-high) × close^5)`**. A nonlinear price-location indicator that uses high powers to amplify extreme cases. |
| `stock_020` | **`(close/prev_close - 1) × volume`**. Daily return multiplied by volume, the simplest capital-flow indicator. |
| `stock_021` | **EMA percentage deviation of intraday range**. `(high-low - EMA(high-low)) / EMA(high-low) × 100` measures volatility's deviation from its smoothed trend. A high value indicates abnormally high volatility that day. |
| `stock_022` | **`corr(average volume, low, 5) + (high+low)/2 - close`**. The correlation between volume and the low plus the midpoint's deviation from the close, combining liquidity and price location. |

---
## extra Family (14 Factors)

This family focuses on features related to trading value (`amount/turnover`) and intraday-structure signals. Trading value is one of the most direct measures of
market participation and liquidity.

| Factor | Idea and Rationale |
|--------|--------------------|
| `extra_001` | **1-day change in close location**. The day-to-day change in the intraday balance between bulls and bears; a positive value indicates stronger bullish control. |
| `extra_002` | **Rank of VWAP-spread extremes + minimum rank × volume-change rank**. Combines the upper and lower extremes of VWAP deviation with volume change in a multidimensional interaction signal. |
| `extra_003` | **Overnight gap (raw value)**. `open/prev_close - 1`, the most basic measure of an overnight information shock. |
| `extra_004` | **AD-oscillator variant**. Fast EMA minus slow EMA of `volume × close_loc`, analogous to applying MACD to capital flow. A fast-line crossover above the slow line is a buy signal. |
| `extra_005` | **Trading-value surge ratio**. `amount / MA20(amount) - 1` measures current trading value's deviation from its 20-day average. A high value indicates an abnormal inflow of capital. |
| `extra_006` | **Cross-sectional rank of relative volume**. The cross-sectional rank of `volume / MA20(volume)` measures the stock's volume activity relative to the market that day. |
| `extra_007` | **Volatility expansion ratio**. Short-term EMA of the intraday range divided by long-term EMA of the range, using the same logic as `better_026`. A ratio greater than 1 indicates expanding volatility. |
| `extra_008` | **Upward-strength/downward-strength ratio (asymmetric windows)**. The 5-day cumulative gain divided by the 10-day cumulative loss. The asymmetric windows allow short-term advances to be reflected in the factor more quickly. |
| `extra_009` | **Conditional volatility spread**. Computes EMAs of volatility only on down days and subtracts the long-term EMA from the short-term EMA. A positive value indicates that down-day volatility has recently increased and risk is accumulating. |
| `extra_010` | **5-day momentum (price ratio)**. `close / delay(close, 5)`, the most basic 5-day price-ratio momentum measure. |
| `extra_011` | **Raw trading value**. Directly uses `amount` or `vwap × volume` as a baseline liquidity measure. |
| `extra_012` | **High/open ratio**. `high / open` measures upside movement after the open. A high ratio indicates strong intraday buying pressure. |
| `extra_013` | **VWAP/close ratio**. `vwap / close`; a value greater than 1 means the intraday average price exceeds the close (more trading occurred at higher intraday prices). |
| `extra_014` | **Intraday range (normalized)**. `(high - low) / close`, the most basic measure of intraday volatility. |

---

## add Family (30 Factors)

Supplementary factor variants covering combinations of momentum, volatility, price-volume relationships, technical patterns, and other dimensions.
They are designed to complement the other families and increase diversity in the factor library.

| Factor | Idea and Rationale |
|--------|--------------------|
| `add_001` | **Price-change rank × volume-change rank**. The product of the cross-sectional ranks of 5-day price change and volume change, signaling a simultaneous price-and-volume advance. |
| `add_002` | **Negative MA deviation × volume rank**. Stocks below their 10-day MA (oversold) with high volume ranks receive high scores; high-volume oversold conditions may rebound. |
| `add_003` | **`sign(Δclose) × (1+\|Δclose/close\|) × Δvolume/volume`**. The direction of price change multiplied by its magnitude and by the rate of volume change, providing three-way signal confirmation. |
| `add_004` | **Volume time-series rank × close-location time-series rank**. Stocks with the highest volume rank over 20 days and highest close-location rank over 10 days are advancing on strong volume. |
| `add_005` | **Up days - down days over 12 days**. The net number of up days is a simple measure of directional persistence. A positive value means up days have clearly outnumbered down days recently. |
| `add_006` | **Short-term momentum EMA - medium-term momentum**. `EMA(daily return, 1/12) - 12-day return`; short-term momentum exceeding medium-term momentum indicates an accelerating trend. |
| `add_007` | **5-day correlation between `rank(close)` and `rank(volume)`**. The short-term correlation between cross-sectional price and volume ranks measures whether the market is chasing high-price, high-volume stocks. |
| `add_008` | **20-day return volatility**. The 20-day standard deviation of daily returns, a classic realized-volatility measure. |
| `add_009` | **3-day change in midpoint price**. The 3-day change in `(high+low)/2` uses the midpoint to avoid noise from extreme prices. |
| `add_010` | **5-day high-low spread / close**. `(5-day highest high - 5-day lowest low) / close` measures the recent price range relative to the current price, a proxy for implied volatility. |
| `add_011` | **10-day correlation between VWAP and volume**. A positive correlation means higher VWAP accompanies higher volume, signaling aggressive institutional buying. |
| `add_012` | **`rank(open deviation from MA10) × rank(\|close-vwap\|)`**. The rank of the open's deviation from trend multiplied by the rank of the close-VWAP deviation. |
| `add_013` | **`(high×low)^0.5 - vwap`**. The difference between the geometric mean price and VWAP (the same as `old_041`), measuring skewness in the intraday price distribution. |
| `add_014` | **5-day return (normalized)**. `(close - delay(close,5)) / delay(close,5)`, basic 5-day momentum. |
| `add_015` | **Cross-sectional rank of overnight gap**. The cross-sectional rank of `open/prev_close - 1` (the same as `best_011`). |
| `add_016` | **`ts_max(vwap,10) - vwap`**. VWAP's distance below its 10-day high, measuring negative momentum/drawdown depth. |
| `add_017` | **`vwap - ts_min(vwap,10)`**. VWAP's distance above its 10-day low, measuring positive momentum/rebound magnitude. |
| `add_018` | **5-day mean of `\|close-open\|/(high-low)`**. The mean real-body fraction: a value near 1 indicates large candlestick bodies (clear trends), while a value near 0 indicates long shadows (indecision). |
| `add_019` | **5-day bullish-candle ratio**. The fraction of the last 5 days on which the close exceeded the open, a short-term measure of bullish sentiment. |
| `add_020` | **Coefficient of variation (10 days)**. `std(close,10) / mean(close,10)` measures relative price volatility; a low CV indicates stable prices, while a high CV indicates greater uncertainty. |
| `add_021` | **`rank(corr(close, volume, 20))`**. The cross-sectional rank of the 20-day correlation between close and volume. A positive long-term price-volume correlation may confirm a trend. |
| `add_022` | **Fast/slow volume EMA ratio**. `EMA(volume, 2/6) / EMA(volume, 2/24)`, analogous to applying MACD to volume. A ratio greater than 1 indicates expanding short-term volume. |
| `add_023` | **`(close - MA5) / close`**. The 5-day MA-deviation ratio; a positive value means price is above its 5-day MA. |
| `add_024` | **`(close - MA20) / close`**. The 20-day MA-deviation ratio; a positive value means price is above its 20-day MA. |
| `add_025` | **`MA5 / MA20`**. The short-term/long-term MA ratio, a continuous-valued version of a golden cross (>1) or death cross (<1). |
| `add_026` | **Volume coefficient of variation (5 days)**. `std(volume,5) / mean(volume,5)` measures volume stability. A high CV means volume fluctuates sharply and may indicate abnormal trading behavior. |
| `add_027` | **`rank(-Δclose × Δvolume)`**. The cross-sectional rank of negative 3-day price change multiplied by 3-day volume change, signaling a low-volume decline (supply contraction). |
| `add_028` | **`(close - open) / (high - low)`**. The day's candlestick "body-direction ratio." +1 is a perfect bullish candle, -1 is a perfect bearish candle, and 0 is a doji. |
| `add_029` | **10-day mean body-direction ratio**. A smoothed version of `add_028` that measures the consistent direction of recent candlestick patterns. |
| `add_030` | **`rank(5-day cumulative return) × rank(20-day cumulative return)`**. The product of short- and medium-term momentum ranks, providing dual-horizon momentum confirmation. |

---
## change Family (5 Factors)

This family focuses on changes in price/volume "change" (second derivatives/acceleration), capturing momentum turning points and regime shifts.

| Factor | Idea and Rationale |
|--------|--------------------|
| `change_001` | **Daily return - 5-day mean return (mean-reversion proxy)**. Measures how far the current daily return deviates from its recent average. A large positive value indicates an abnormal one-day rise that may revert; a large negative value may indicate an oversold rebound opportunity. |
| `change_002` | **Price acceleration (`Δ²close`)**. The second difference of the closing price normalized by price measures whether momentum is accelerating or decelerating. A transition from positive to negative may be an early sign of a trend top. |
| `change_003` | **Rank of volume deviation from its mean**. The cross-sectional rank of `(volume - MA20(volume))` measures how abnormal the stock's current volume is relative to both its own history and the overall market. |
| `change_004` | **Range expansion ratio**. `(high-low) / MA20(high-low) - 1` measures the current intraday range's deviation from its recent average. A high value indicates abnormally high volatility that day, potentially a breakout or panic. |
| `change_005` | **5-day directional sum**. The 5-day sum of `sign(Δclose)`. +5 means five consecutive up days and -5 means five consecutive down days. Extreme values indicate a very strong trend or an imminent reversal. |

---

## original Family (28 Factors)

The most basic price-volume statistical factors, constructed directly from closing prices and volume with minimal transformation. These factors serve as
"atomic components" for more complex factors while also being effective standalone signals.

| Factor | Idea and Rationale |
|--------|--------------------|
| `original_001` | **20-day volatility (cross-sectional z-score)**. The 20-day standard deviation of daily returns, standardized cross-sectionally. A high value means the stock is significantly more volatile than the market average, implying high risk/high expected return. |
| `original_002` | **Rank of distance from the 20-day high**. The cross-sectional rank of `close / 20-day highest close - 1`. A value near 0 indicates proximity to the high, while a large negative value indicates a substantial drawdown. |
| `original_003` | **Rank of distance from the 20-day low**. The cross-sectional rank of `close / 20-day lowest close - 1`. A high value indicates a substantial rebound from the bottom. |
| `original_004` | **60-day time-series rank of close**. The current close's relative position over the last 60 days. A high rank indicates that price is near a quarterly high. |
| `original_005` | **60-day time-series rank of volume**. Current volume's relative position over the last 60 days. A high rank indicates abnormally active trading that day. |
| `original_006` | **20-day price-volume correlation**. The 20-day rolling correlation between close and volume. A positive correlation means price and volume move together (trend confirmation). |
| `original_007` | **20-day time-series rank of close**. The current close's position over the last 20 days, representing monthly momentum. |
| `original_008` | **20-day time-series rank of volume**. Current volume's position over the last 20 days, representing monthly trading activity. |
| `original_009` | **10-day price-volume correlation**. The short-term correlation between close and volume. |
| `original_010` | **5-day price-volume correlation**. The shortest-window price-volume relationship, most sensitive to recent changes. |
| `original_011` | **10-day time-series rank of close**. Price location on a roughly two-week horizon. |
| `original_012` | **10-day time-series rank of volume**. Volume activity on a roughly two-week horizon. |
| `original_013` | **5-day time-series rank of close**. Price location on a one-week horizon. |
| `original_014` | **5-day time-series rank of volume**. Volume activity on a one-week horizon. |
| `original_015` | **10-day volatility**. The 10-day standard deviation of daily returns, representing approximately two weeks of realized volatility. |
| `original_016` | **5-day volatility**. The 5-day standard deviation of daily returns, representing one week of realized volatility and responding most strongly to recent volatility changes. |
| `original_017` | **Distance from the 10-day high**. `close / 10-day highest high - 1`, a raw value (not a rank) that directly measures drawdown depth. |
| `original_018` | **Distance from the 10-day low**. `close / 10-day lowest low - 1` directly measures rebound magnitude. |
| `original_019` | **Distance from the 5-day high**. `close / 5-day highest high - 1` measures drawdown depth within one week. |
| `original_020` | **Distance from the 5-day low**. `close / 5-day lowest low - 1` measures rebound magnitude within one week. |
| `original_021` | **10-day momentum**. `close / delay(close,10) - 1`, the two-week rate of price change. |
| `original_022` | **5-day momentum**. `close / delay(close,5) - 1`, the one-week rate of price change. |
| `original_023` | **3-day momentum**. `close / delay(close,3) - 1`, an ultra-short-term trend. |
| `original_024` | **1-day momentum (daily return)**. `close / delay(close,1) - 1`, the most basic daily return. |
| `original_025` | **Cross-sectional rank of 10-day momentum**. The cross-sectional relative position of two-week momentum, indicating outperformance or underperformance versus the market. |
| `original_026` | **Cross-sectional rank of 5-day momentum**. The cross-sectional relative position of one-week momentum. |
| `original_027` | **Cross-sectional rank of 1-day momentum**. The cross-sectional relative position of the daily return: where the stock ranks in the entire market today. |
| `original_028` | **Cross-sectional rank of 20-day momentum**. The cross-sectional relative position of monthly momentum, providing a monthly strength ranking. |

---

## cs_rank Family (6 Factors)

Market-breadth signals. Each stock's daily change (relative to the previous close) is ranked cross-sectionally,
providing a standardized measure of "where this stock's performance ranks in the entire market today."

| Factor | Idea and Rationale |
|--------|--------------------|
| `cs_rank_close` | **Cross-sectional rank of close return**. The cross-sectional rank of `(close/prev_close - 1)`, the most direct measure of today's relative strength. |
| `cs_rank_open` | **Cross-sectional rank of gap size**. The cross-sectional rank of `(open/prev_close - 1)`, measuring the relative strength of the overnight information shock. |
| `cs_rank_high` | **Cross-sectional rank of intraday high return**. The cross-sectional rank of `(high/prev_close - 1)`, ranking intraday upside strength. |
| `cs_rank_low` | **Cross-sectional rank of intraday low return**. The cross-sectional rank of `(low/prev_close - 1)`, ranking intraday downside depth. |
| `cs_rank_avg` | **Cross-sectional rank of VWAP return**. The cross-sectional rank of `(vwap/prev_close - 1)`, measuring the relative strength of the institutional average transaction price versus the previous close. |
| `cs_rank_amount` | **Cross-sectional rank of trading value**. The cross-sectional rank of `amount`, a relative measure of market attention/liquidity that day. |

---

## Factor Selection Recommendations

### Selection by Strategy Type

| Strategy Type | Recommended Factor Families | Rationale |
|---------------|-----------------------------|-----------|
| Short-term reversal | `change_*`, `best_001-005`, `old_049/051` | Capture oversold/overbought extremes |
| Trend following | `original_021-028`, `add_005/006`, `best_017-019` | Multi-horizon momentum |
| Price-volume relationship | `old_028-030`, `add_007/011/021`, `stock_001-002` | Price-volume synchronization/divergence |
| Liquidity | `better_008/020`, `extra_005/006`, `stock_010/015` | Amihud and abnormal trading value |
| Volatility | `original_001/015/016`, `add_008/020`, `change_004` | Full spectrum of realized volatility |
| Institutional behavior | `better_*`, `best_008-010`, `cs_rank_avg` | VWAP-deviation family |

### Factor Correlation Notes

- Most factors within `original_*` are highly correlated (the same indicator over different windows); select 1-2 windows from each category.
- `better_*` and `best_*` overlap partially (both use VWAP), but approach it from different perspectives and can coexist.
- The `old_*` family has the greatest internal diversity and the lowest correlations, making it suitable for use in full.
- `add_*` mixes components from other families and works better after removing duplicates with factors already selected.

---

*This handbook is updated along with the factor library. To add custom factors, see [docs/factors.md](factors.md#adding-a-new-factor).*
