# AllData & AllModel

Every model on the full 2016–2025 data, with and without the market (SPY).

- **Notebook:** `AllData_AllModel.ipynb` (built by `tools/make_notebook.py`). Run on Colab with a GPU (L4 or A100).
- **Data:** `alldata_bundle_v1.zip`, built locally by `tools/build_alldata_bundle.py`. It holds licensed WRDS
  data, so it stays on Drive and is never committed.
- **Code:** `chartcnn/`, a copy of `CNN-Relevant/chartcnn` extended with SPY inputs (version `2026-11-16.1`).

## Data

| | |
|---|---|
| Universe | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 members, by PERMNO: 775 securities, 516–531 a day |
| Sessions | 2,514 from 2016-01-04 to 2025-12-31; the 21 early-close sessions are excluded from windows |
| Bars | Five-minute OHLCV from TAQ trades, regular session, filters as in `US_EQUITY_ARCHIVE.md` |
| Market | SPY (PERMNO 84398) built with the same filters on the same clock |
| Adjustment | CRSP CIZ factors and cash dividends; each window is put on the basis of its own last bar |
| Split | train 2016–2022, validation 2023, test 2024–2025 |

Build steps, from raw archive to bundle:

```bash
python prepare_archive_inputs.py                      # CIZ -> adjustment inputs (stocks)
python adjust_dividend_bars.py --raw-root ..._adjustment_inputs --inputs ..._adjustment_inputs --output ..._dividend_adjusted
python validate_dividend_bars.py --root ..._dividend_adjusted
python download_benchmark_bars.py                     # CV repository: SPY bars on the same calendar
# same three steps for the SPY archive
python -m us_pipeline prepare --config configs/us_5min_2016_2025.json
python -m us_pipeline audit   --config configs/us_5min_2016_2025.json
python "AllData & AllModel/tools/build_alldata_bundle.py" --prepared ... --market ... --output .../alldata_bundle_v1.zip
```

**Distribution classification (CIZ).** A record with a cash amount and no price or share factor is a cash
dividend and gets the proportional chart adjustment. This covers ordinary, special, return of capital, capital
gains and non-USD dividends. A record with a factor is covered by the cumulative factors alone. 19,374 of
19,384 cash ex-date groups match CRSP's own `ret - retx` within 1%. That includes the 8 non-USD dividends of
CCEP and FER, whose CIZ amounts are already in USD; all 8 match to four decimals. The 10 that do not match are
return-of-capital / capital-gain mixes and one security exchange, where CRSP's `retx` treats part of the
payment differently. The price still drops by the full amount.

## Models per cell

| name | input | parameters |
|---|---|---|
| `plain` | stock chart 234×234 | 776,450 |
| `plain_spy` | stock chart over SPY chart, 468×234, bar columns aligned | 936,194 |
| `series1d` | 5 scaled OHLCV series | 720,770 |
| `series1d_spy` | 10 series: stock and SPY | 723,970 |
| HGB hand features | 20 features of the stock window | tree nodes, printed per cell |
| HGB hand + SPY | plus 18 SPY and stock-versus-SPY features | tree nodes, printed per cell |

Targets: `dir`, `vol` and `xsdir` at 5, 15, 30 and 60 minutes, stride 12.
