import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
        ax.set_xticklabels(areas, rotation=45, ha='right')
        
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
        print(f"Successfully loaded {len(df_raw)} total runs across {df_raw['area'].nunique() if 'area' in df_raw.columns else 1} areas.")
        plot_with_matplotlib(df_raw, graphs_dir)
        print_insights(df_raw)
        print(f"\nAll graphs have been saved to: {graphs_dir}")
    else:
        print("Could not generate plots because no data was found.")
