import os
import glob
import pandas as pd
from itertools import combinations
from scipy.stats import ks_2samp


def load_all_results(results_dir: str) -> pd.DataFrame:
    search_pattern = os.path.join(results_dir, '*_summary.xlsx')
    files = glob.glob(search_pattern)

    if not files:
        all_res = os.path.join(results_dir, 'summary_results.xlsx')
        if os.path.exists(all_res):
            files = [all_res]
        else:
            print(f"No *_summary.xlsx files found in {results_dir}")
            return pd.DataFrame()

    df_list = [pd.read_excel(f) for f in files]
    print(f'length of samples - {len(df_list)}')
    return pd.concat(df_list, ignore_index=True)


def run_ks_tests(df_raw: pd.DataFrame):
    mode_mapping = {
        'baseline':           'SA',
        'exact_cache':        'SA + Exact Cache',
        'intelligent_memory': 'SA + LTMB',
    }

    df = df_raw.copy()
    df['mode'] = df['mode'].replace(mode_mapping)

    approaches = ['SA', 'SA + Exact Cache', 'SA + LTMB']

    metrics = {
        'obj_mean':        'Objective Function Value (obj_mean)',
        'time_s_mean':     'Execution Time (time_s_mean)',
        '%_covered_mean':  'Edge Coverage (% covered_mean)',
    }

    for col, label in metrics.items():
        print("\n" + "=" * 60)
        print(f"KS TEST RESULTS — {label}")
        print("=" * 60)

        for a, b in combinations(approaches, 2):
            x = df[df['mode'] == a][col].dropna().values
            y = df[df['mode'] == b][col].dropna().values

            if len(x) == 0 or len(y) == 0:
                print(f"\n{a} vs {b}: insufficient data (n={len(x)}, n={len(y)})")
                continue

            stat, p = ks_2samp(x, y)
            sig = "*** SIGNIFICANT" if p < 0.05 else "not significant"
            print(f"\n{a}  vs  {b}")
            print(f"  KS statistic : {stat:.4f}")
            print(f"  p-value      : {p:.4e}  [{sig}]")
            print(f"  n({a}) = {len(x)},  n({b}) = {len(y)}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'routing_results')

    print(f"Scanning for results in: {results_dir}")
    df_raw = load_all_results(results_dir)

    if not df_raw.empty:
        n_areas = df_raw['area'].nunique() if 'area' in df_raw.columns else 1
        print(f"Loaded {len(df_raw)} rows across {n_areas} areas.")
        run_ks_tests(df_raw)
    else:
        print("No data found — cannot run KS tests.")
