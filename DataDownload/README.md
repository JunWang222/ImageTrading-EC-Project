# DataDownload: TAQ bars and reference data from WRDS

Step 1 of the data pipeline (see the root README for the whole order). All scripts read WRDS. The user
name is `$WRDS_USERNAME`, or the local login name; the password comes from pgpass and is never stored here.
Run each script from this folder. They import each other by module name.

| Script | What it writes |
|---|---|
| `download_us_equity_archive.py` | 2016-onward archive: universe, TAQ mappings, raw 5-minute bars (one partition per session), CRSP CIZ corporate actions and daily reference. See `US_EQUITY_ARCHIVE.md`. Single-threaded and resumable: about 30 s per session, about 23 h for 2016-2025. |
| `run_archive_job.py OUTPUT` | Runs the archive downloader unattended and writes `00_metadata/job_exit.json`. |
| `download_benchmark_bars.py` | SPY (or `--symbol`) on the stock archive's calendar, with the same filter and grid. |
| `download_intraday_bars.py` | The trade filter and aggregation SQL (`aggregate_sql`, `POLICY`) that the archive and benchmark import, plus the 2023 H1 demo downloader. See `INTRADAY_DEMO.md`. |
| `build_dynamic_universe.py` | Daily S&P 500 ∪ NASDAQ 100 ∪ DJIA 30 membership by PERMNO. |
| `fetch_distribution_inputs.py`, `inspect_wrds_demo.py` | Demo-only: dividend inputs and a WRDS schema probe. |

**Trade filter, policy version 3:**

```sql
price > 0 AND size > 0 AND tr_corr = '00'
AND translate(coalesce(tr_scond, ''), ' @EFI', '') = ''
```

A blank sale condition is a regular sale. WRDS stores it as NULL, so NULL counts as blank. `I` marks an odd lot.
Versions 1 and 2 are wrong for model data:
- version 1 dropped NULL;
- version 2 dropped odd lots.

The evidence is in `../DataCheck/`. Check every new download there before using it.

```bash
cd DataDownload
pip install -r requirements-archive.txt
# pilot: one session per period, then run the DataCheck scripts on it
python -u download_us_equity_archive.py --output ~/"Market Data/US Equity/v3/01_raw" --limit-days 1
# full archive (resumes after an interruption)
python run_archive_job.py ~/"Market Data/US Equity/v3/01_raw"
python download_benchmark_bars.py --stock-archive ~/"Market Data/US Equity/v3/01_raw" \
       --output ~/"Market Data/US Equity/v3/01_raw_spy"
```
