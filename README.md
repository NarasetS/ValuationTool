# Stock Valuation Sandbox

A Streamlit valuation app that pulls public market data with `yfinance` and evaluates a ticker with:

- Relative valuation against an editable peer group
- A 10-year DCF model using both exit multiple and perpetuity terminal value methods
- Enterprise value to equity value bridge using cash, debt, and shares outstanding
- Manual fallback inputs when `yfinance` data is missing or stale

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run

```powershell
streamlit run app.py
```

## Test

```powershell
python -m pytest
```

## Model Notes

- Free cash flow is calculated as operating cash flow minus the absolute value of capital expenditures.
- The DCF starts from a representative annual free cash flow baseline and projects 10 annual cash flows using the Stage 1 growth rate.
- The app offers four baseline choices: TTM FCF from the latest four quarters, latest fiscal year FCF, 4-year average annual FCF, and manual normalized FCF.
- Manual normalized FCF is often the cleanest choice when recent working capital movements, one-time capex, commodity cycles, restructurings, or acquisition effects distort reported FCF.
- Enterprise value equals the present value of projected free cash flows plus the present value of terminal value.
- Equity value equals enterprise value plus total cash minus total debt.
- Intrinsic value per share equals equity value divided by diluted shares outstanding.
- Perpetuity terminal value is disabled when the perpetuity growth rate is greater than or equal to the discount rate.

## Data Caveats

`yfinance` is a convenient public data source, but fields can be missing, delayed, restated, or mapped differently across issuers. The app warns when critical values are missing and exposes manual input boxes for free cash flow, cash, debt, and shares outstanding.
