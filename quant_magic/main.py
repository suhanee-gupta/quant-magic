from __future__ import annotations

import argparse
from pathlib import Path

from .backtest import run_backtest
from .var_engine import run_var_analysis

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


def main(run_var: bool = False):
    root = Path(__file__).resolve().parent.parent
    output_dir = root / 'results'
    result = run_backtest(
        SENSEX_30,
        START_DATE,
        END_DATE,
        TRAIN_END,
        VAL_END,
        SENSEX_BENCHMARK,
        output_dir=output_dir,
    )

    if run_var:
        run_var_analysis(
            result['asset_returns'],
            result['weights'],
            result['test_start'],
            results_dir=output_dir,
            plots_dir=root / 'plots',
        )

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the Markowitz backtest.')
    parser.add_argument('--var', action='store_true', help='also run the VaR engine and backtest')
    args = parser.parse_args()
    main(run_var=args.var)
