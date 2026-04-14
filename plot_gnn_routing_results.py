import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

def load_all_results(results_dir: str) -> pd.DataFrame:
    """
    Finds all `*_results.xlsx` files in the subdirectories of results_dir
    and concatenates them into a single global DataFrame.
    """
    search_pattern = os.path.join(results_dir, '*_summary.xlsx')
    files = glob.glob(search_pattern)
    
    if not files:
        # Also check if there's a global summary_results.xlsx
        all_res = os.path.join(results_dir, 'summary_results.xlsx')
        if os.path.exists(all_res):
            files = [all_res]
        else:
            print(f"No *_summary.xlsx files found in {results_dir}")
            return pd.DataFrame()
            
    df_list = [pd.read_excel(f) for f in files]
    df_all = pd.concat(df_list, ignore_index=True)
    return df_all

def plot_with_matplotlib(df_raw: pd.DataFrame, out_dir: str):
    """
    Uses vanilla matplotlib to plot grouped bar charts with error bars (standard deviation).
    """
    mode_mapping = {
        'baseline': 'SA',
        'exact_cache': 'SA + Exact Cache',
        'intelligent_memory': 'SA + LTMB'
    }
    
    df = df_raw.copy()
    if 'mode' in df.columns:
        df['mode'] = df['mode'].replace(mode_mapping)
        
    if 'area' in df.columns:
        # Clean area names (e.g. 'Little_Rock_Arkansas_US' -> 'Little Rock')
        df['area_clean'] = df['area'].apply(lambda x: x.replace('_', ' ').split(',')[0] if isinstance(x, str) else x)
    else:
        print("Missing 'area' column in data!")
        return

    metrics = {
        'obj_mean': ('Objective Function Value', 'Objective', 'obj_comparison.png'),
        'time_s_mean': ('Execution Time in Seconds', 'Time (s)', 'time_comparison.png'),
        '%_covered_mean': ('Percentage of Edges Covered', 'Coverage (%)', 'coverage_comparison.png')
    }

    os.makedirs(out_dir, exist_ok=True)

    # Reorder columns to display intelligently
    hue_order = ['SA', 'SA + Exact Cache', 'SA + LTMB']
    modes_present = [m for m in hue_order if m in df['mode'].unique()]
    areas = sorted(df['area_clean'].unique())
    
    # Lighter, professional academic palette (Set2-like)
    colors = ['#8da0cb', '#fc8d62', '#66c2a5']
    
    for col, (title, ylabel, filename) in metrics.items():
        if col not in df.columns:
            print(f"Warning: Column {col} not found in results dataframe.")
            continue
            
        fig, ax = plt.subplots(figsize=(14, 6))
        
        x = np.arange(len(areas))
        # Total width depends on number of modes. E.g., if 3 modes, width = 0.25
        width = 0.8 / len(modes_present) if len(modes_present) > 0 else 0.8
        multiplier = 0
        
        for i, mode in enumerate(modes_present):
            mode_data = df[df['mode'] == mode]
            
            means = []
            stds = []
            for a in areas:
                area_data = mode_data[mode_data['area_clean'] == a][col]
                if not area_data.empty:
                    means.append(area_data.mean())
                    stds.append(area_data.std() if len(area_data) > 1 else 0)
                else:
                    means.append(0)
                    stds.append(0)
                    
            # Calculate x offset based on multiplier
            offset = width * multiplier
            ax.bar(x + offset, means, width, label=mode, yerr=stds, capsize=4,
                   color=colors[i % len(colors)], edgecolor='black', linewidth=1, error_kw={'elinewidth': 1.5, 'alpha': 0.8})
            multiplier += 1
            
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.set_xlabel('City / Read-World Graph Instance', fontsize=12, labelpad=10)
        ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        
        # Center the xticks
        ax.set_xticks(x + width * (len(modes_present) - 1) / 2)
        ax.set_xticklabels(areas, rotation=0, ha='right')
        
        # Adjust legend
        ax.legend(title='Routing Approach', bbox_to_anchor=(1.01, 1), loc='upper left')
        
        # Add light horizontal grid lines behind the bars
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        save_path = os.path.join(out_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {save_path}")

# ---------------------------------------------------------------------------
# Helper: build a per-city summary DataFrame (SA as baseline)
# ---------------------------------------------------------------------------
def _build_city_summary(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates obj_mean, time_s_mean, %_covered_mean per (area_clean, mode).
    Returns a wide-format DataFrame indexed by city with columns for each metric/mode.
    """
    mode_mapping = {
        'baseline': 'SA',
        'exact_cache': 'Exact Cache',
        'intelligent_memory': 'LTMB'
    }
    df = df_raw.copy()
    df['mode'] = df['mode'].replace(mode_mapping)
    df['area_clean'] = df['area'].apply(
        lambda x: x.split(',')[0].replace('_', ' ').strip() if isinstance(x, str) else str(x)
    )
    # Average over periods per city+mode
    grp = df.groupby(['area_clean', 'mode'])[['obj_mean', 'time_s_mean', '%_covered_mean']].mean().reset_index()
    return grp


# ---------------------------------------------------------------------------
# Plot 1b: Runtime Reduction (%) relative to SA baseline
# ---------------------------------------------------------------------------
def plot_runtime_reduction(df_raw: pd.DataFrame, out_dir: str):
    """
    Bar chart showing % speed-up of Exact Cache and LTMB vs SA for each city.
    Formula: (T_SA - T_Method) / T_SA * 100
    """
    grp = _build_city_summary(df_raw)
    os.makedirs(out_dir, exist_ok=True)

    sa_data    = grp[grp['mode'] == 'SA'][['area_clean', 'time_s_mean']].set_index('area_clean')
    cache_data = grp[grp['mode'] == 'Exact Cache'][['area_clean', 'time_s_mean']].set_index('area_clean')
    ltmb_data  = grp[grp['mode'] == 'LTMB'][['area_clean', 'time_s_mean']].set_index('area_clean')

    cities = sorted(sa_data.index)

    def pct_reduction(method_df):
        vals = []
        for c in cities:
            if c in sa_data.index and c in method_df.index:
                t_sa  = sa_data.loc[c, 'time_s_mean']
                t_m   = method_df.loc[c, 'time_s_mean']
                vals.append((t_sa - t_m) / (t_sa + 1e-9) * 100)
            else:
                vals.append(np.nan)
        return np.array(vals)

    cache_pct = pct_reduction(cache_data)
    ltmb_pct  = pct_reduction(ltmb_data)

    x = np.arange(len(cities))
    width = 0.45

    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width / 2, cache_pct, width, label='SA + Exact Cache',
                   color='#fc8d62', edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width / 2, ltmb_pct,  width, label='SA + LTMB',
                   color='#66c2a5', edgecolor='black', linewidth=0.8)

    # Annotate each bar — padding=3 is in points, so gap is figsize-independent
    ax.bar_label(bars1, fmt='%.1f%%', padding=2, fontsize=11, fontweight='bold')
    ax.bar_label(bars2, fmt='%.1f%%', padding=2, fontsize=11, fontweight='bold')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_title('Runtime Reduction vs SA Baseline', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('City / Real-World Graph Instance', fontsize=12, labelpad=10)
    ax.set_ylabel('Speedup over SA (%)', fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(cities, rotation=0, ha='right')
    ax.legend(title='Routing Approach', loc='upper left')
    ax.set_axisbelow(True)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'runtime_reduction_pct.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 2: Quality vs Time Trade-off Scatter (Pareto view)
# ---------------------------------------------------------------------------
def plot_quality_vs_time(df_raw: pd.DataFrame, out_dir: str):
    """
    Scatter plot: X = execution time, Y = objective value.
    Each point is a (city, mode) pair. Shows Pareto trade-off.
    """
    grp = _build_city_summary(df_raw)
    os.makedirs(out_dir, exist_ok=True)

    mode_styles = {
        'SA':          {'color': '#8da0cb', 'marker': 'o', 'zorder': 2},
        'Exact Cache': {'color': '#fc8d62', 'marker': 's', 'zorder': 3},
        'LTMB':        {'color': '#66c2a5', 'marker': '^', 'zorder': 4},
    }

    fig, ax = plt.subplots(figsize=(9, 7))

    for mode, style in mode_styles.items():
        sub = grp[grp['mode'] == mode]
        ax.scatter(
            sub['time_s_mean'], sub['obj_mean'],
            c=style['color'], marker=style['marker'],
            s=120, edgecolors='black', linewidths=0.7,
            label=f'SA + {mode}' if mode != 'SA' else 'SA (Baseline)',
            zorder=style['zorder']
        )
        # Label each point with city name
        for _, row in sub.iterrows():
            ax.annotate(
                row['area_clean'],
                (row['time_s_mean'], row['obj_mean']),
                textcoords='offset points', xytext=(5, 4),
                fontsize=7, alpha=0.85
            )

    ax.set_title('Quality vs. Execution Time Trade-off', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('Execution Time (s)', fontsize=12, labelpad=10)
    ax.set_ylabel('Objective Function Value (↑ better)', fontsize=12, labelpad=10)

    # Pareto annotation arrow
    ax.annotate(
        'Better\n(faster & higher quality)',
        xy=(0.05, 0.95), xycoords='axes fraction',
        fontsize=9, color='#2d6a4f', fontstyle='italic',
        ha='left', va='top'
    )

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#8da0cb',
               markersize=10, markeredgecolor='black', label='SA (Baseline)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#fc8d62',
               markersize=10, markeredgecolor='black', label='SA + Exact Cache'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#66c2a5',
               markersize=10, markeredgecolor='black', label='SA + LTMB'),
    ]
    ax.legend(handles=legend_elements, title='Method', loc='lower right')
    ax.set_axisbelow(True)
    ax.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'quality_vs_time_scatter.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 3: Scalability Plot (Runtime vs Graph Complexity)
# ---------------------------------------------------------------------------
def plot_scalability(df_raw: pd.DataFrame, out_dir: str):
    """
    Line plot: X = graph complexity rank (cities sorted by SA runtime as proxy
    for graph size), Y = mean execution time. One line per method.

    Note: If explicit |E| or |N| data is available in df_raw, replace
    'sa_runtime_rank' with the actual graph-size column.
    """
    grp = _build_city_summary(df_raw)
    os.makedirs(out_dir, exist_ok=True)

    # Sort cities by SA execution time as a proxy for graph complexity
    sa_times = (
        grp[grp['mode'] == 'SA']
        .set_index('area_clean')['time_s_mean']
        .sort_values()
    )
    city_order = sa_times.index.tolist()  # Ascending complexity

    mode_styles = {
        'SA':          {'color': '#8da0cb', 'ls': '-',  'marker': 'o'},
        'Exact Cache': {'color': '#fc8d62', 'ls': '--', 'marker': 's'},
        'LTMB':        {'color': '#66c2a5', 'ls': '-.',  'marker': '^'},
    }

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(city_order))

    for mode, style in mode_styles.items():
        sub = grp[grp['mode'] == mode].set_index('area_clean')
        times = [sub.loc[c, 'time_s_mean'] if c in sub.index else np.nan for c in city_order]
        label = f'SA + {mode}' if mode != 'SA' else 'SA (Baseline)'
        ax.plot(
            x, times,
            color=style['color'], linestyle=style['ls'],
            marker=style['marker'], markersize=8,
            linewidth=2.2, label=label
        )

    ax.set_title('Scalability: Execution Time vs. Graph Complexity', fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel('City (ordered by increasing graph complexity → SA runtime)', fontsize=12, labelpad=10)
    ax.set_ylabel('Mean Execution Time (s)', fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(city_order, rotation=0, ha='right')
    ax.legend(title='Method', loc='upper left')
    ax.set_axisbelow(True)
    ax.grid(linestyle='--', alpha=0.6)
    plt.tight_layout()
    save_path = os.path.join(out_dir, 'scalability_plot.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


# ---------------------------------------------------------------------------
# Plot 4: Delta Plot (Method - SA baseline per city)
# ---------------------------------------------------------------------------
def plot_delta_from_baseline(df_raw: pd.DataFrame, out_dir: str):
    """
    Two-panel bar chart showing:
      top    panel: Δ Objective  = Method_obj  - SA_obj
      bottom panel: Δ Coverage   = Method_cov  - SA_cov
    Centre line at 0. Values ≥ 0 mean no degradation vs SA.
    """
    grp = _build_city_summary(df_raw)
    os.makedirs(out_dir, exist_ok=True)

    sa = grp[grp['mode'] == 'SA'].set_index('area_clean')
    methods = {'Exact Cache': '#fc8d62', 'LTMB': '#66c2a5'}

    cities = sorted(sa.index)
    x = np.arange(len(cities))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    fig.suptitle('Δ Performance vs SA Baseline (Method − SA)', fontsize=16, fontweight='bold', y=1.01)

    for panel_idx, (metric, label, ax) in enumerate([
        ('obj_mean',       'Δ Objective Function Value', ax1),
        ('%_covered_mean', 'Δ Coverage (%)',             ax2),
    ]):
        for i, (mode, color) in enumerate(methods.items()):
            sub = grp[grp['mode'] == mode].set_index('area_clean')
            deltas = []
            for c in cities:
                if c in sa.index and c in sub.index:
                    deltas.append(sub.loc[c, metric] - sa.loc[c, metric])
                else:
                    deltas.append(np.nan)

            offset = (i - 0.5) * width
            bars = ax.bar(
                x + offset, deltas, width,
                label=f'SA + {mode}',
                color=color, edgecolor='black', linewidth=0.7
            )

            for bar, val in zip(bars, deltas):
                if not np.isnan(val):
                    va = 'bottom' if val >= 0 else 'top'
                    y_off = 0.003 * (1 if val >= 0 else -1)
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val + y_off,
                        f'{val:+.2f}',
                        ha='center', va=va, fontsize=7, fontweight='bold'
                    )

        ax.axhline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_ylabel(label, fontsize=11)
        ax.legend(title='Method', loc='upper right')
        ax.set_axisbelow(True)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels(cities, rotation=0, ha='right')
    ax2.set_xlabel('City / Real-World Graph Instance', fontsize=12, labelpad=10)

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'delta_from_baseline.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def print_insights(df_raw: pd.DataFrame):
    """
    Computes statistical highlights for terminal output.
    """
    print("\n" + "="*60)
    print("📈 ROUTING ALGORITHM INSIGHTS & SUMMARY")
    print("="*60)

    mode_mapping = {
        'baseline': 'SA',
        'exact_cache': 'SA + Exact Cache',
        'intelligent_memory': 'SA + LTMB'
    }
    df = df_raw.copy()
    df['mode'] = df['mode'].replace(mode_mapping)

    # Group globally to see overall performance
    global_summary = df.groupby('mode')[['obj_mean', 'time_s_mean', '%_covered_mean']].agg(['mean', 'std']).round(2)
    print("\n--- GLOBAL AGGREGATE PERFORMANCE (Averaged across ALL areas/periods) ---")
    
    # Flatten multiindex for prettier printing
    flat_summary = global_summary.copy()
    flat_summary.columns = [f"{col[0]}_{col[1]}" for col in flat_summary.columns]
    
    print(flat_summary.to_string())

    # Highlight wins
    print("\n--- KEY INSIGHTS ---")
    if 'SA' in df['mode'].unique() and 'SA + LTMB' in df['mode'].unique():
        baseline_time = flat_summary.loc['SA', 'time_s_mean_mean']
        ltm_time = flat_summary.loc['SA + LTMB', 'time_s_mean_mean']
        speedup = (baseline_time - ltm_time) / (baseline_time + 1e-9) * 100
        print(f"1. Execution Time: LTMB is {speedup:.1f}% {'faster' if speedup>0 else 'slower'} than the Baseline on average.")
        
        baseline_obj = flat_summary.loc['SA', 'obj_mean_mean']
        ltm_obj = flat_summary.loc['SA + LTMB', 'obj_mean_mean']
        obj_improv = (ltm_obj - baseline_obj) / (baseline_obj + 1e-9) * 100
        print(f"2. Objective Quality: LTMB improves the objective score by {obj_improv:.1f}% compared to Baseline (Higher is Better).")
        
        baseline_cov = flat_summary.loc['SA', '%_covered_mean_mean']
        ltm_cov = flat_summary.loc['SA + LTMB', '%_covered_mean_mean']
        print(f"3. Coverage: LTMB achieves {ltm_cov:.1f}% coverage vs Baseline's {baseline_cov:.1f}%.")

if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'routing_results')
    graphs_dir = os.path.join(results_dir, 'analysis_graphs')
    
    print(f"Scanning for results in: {results_dir}")
    df_raw = load_all_results(results_dir)
    
    if not df_raw.empty:
        n_areas = df_raw['area'].nunique() if 'area' in df_raw.columns else 1
        print(f"Successfully loaded {len(df_raw)} total runs across {n_areas} areas.")

        # --- Original grouped bar charts ---
        plot_with_matplotlib(df_raw, graphs_dir)

        # --- New PhD-level plots ---
        print("\nGenerating advanced analysis plots...")
        plot_runtime_reduction(df_raw, graphs_dir)
        plot_quality_vs_time(df_raw, graphs_dir)
        plot_scalability(df_raw, graphs_dir)
        plot_delta_from_baseline(df_raw, graphs_dir)

        print_insights(df_raw)
        print(f"\nAll graphs have been saved to: {graphs_dir}")
    else:
        print("Could not generate plots because no data was found.")
