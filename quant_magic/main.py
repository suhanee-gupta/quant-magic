from __future__ import annotations

from pathlib import Path

from .backtest import run_backtest
from .data import fetch_data

SENSEX_30 = [
    'RELIANCE.BO',  'TCS.BO',       'HDFCBANK.BO',  'INFY.BO',      'ICICIBANK.BO',
    'HINDUNILVR.BO','ITC.BO',       'SBIN.BO',      'BHARTIARTL.BO','KOTAKBANK.BO',
    'LT.BO',        'AXISBANK.BO',  'ASIANPAINT.BO','MARUTI.BO',    'SUNPHARMA.BO',
    'TITAN.BO',     'BAJFINANCE.BO','WIPRO.BO',     'ONGC.BO',      'NTPC.BO',
    'POWERGRID.BO', 'ULTRACEMCO.BO','NESTLEIND.BO', 'BAJAJFINSV.BO','TATAMOTORS.NS',
    'HCLTECH.BO',   'TATASTEEL.BO', 'JSWSTEEL.BO',  'M&M.BO',       'ADANIENT.BO',
]
SENSEX_BENCHMARK = '^BSESN'

START_DATE = '2014-01-01'
END_DATE = '2024-12-31'
TRAIN_END = '2021-12-31'
VAL_END = '2022-12-31'


def main():
    output_dir = Path(__file__).resolve().parent.parent / 'results'
    run_backtest(
        SENSEX_30,
        START_DATE,
        END_DATE,
        TRAIN_END,
        VAL_END,
        SENSEX_BENCHMARK,
        output_dir=output_dir,
    )


if __name__ == '__main__':
    main()
