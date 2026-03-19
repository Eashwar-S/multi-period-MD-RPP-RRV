import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def main():
    folder = r"c:\Users\eashw\Desktop\PhD_resesarch\multi-period-MD-RPP-RRV\multi_period_routing\results"
    gurobi_file = os.path.join(folder, "optimisation_results.xlsx")
    sa_file = os.path.join(folder, "summary_results_new.xlsx")

    print("Loading datasets...")
    df_gurobi = pd.read_excel(gurobi_file)
    df_sa = pd.read_excel(sa_file)

    print("Processing Gurobi data...")
    df_g = df_gurobi[['graph_id', 'interval', 'num_nodes', 'obj', 'time_s', '%_covered']].copy()
    df_g.rename(columns={'interval': 'Interval'}, inplace=True)
    df_g['mode'] = 'Gurobi'
    df_g['obj_mean'] = df_g['obj']
    df_g['obj_std'] = 0.0
    df_g['time_s_mean'] = df_g['time_s']
    df_g['time_s_std'] = 0.0
    df_g['%_covered_mean'] = df_g['%_covered']
    df_g['%_covered_std'] = 0.0
    df_g.drop(columns=['obj', 'time_s', '%_covered'], inplace=True)

    print("Processing SA data...")
    graph_info = df_g[['graph_id', 'num_nodes']].drop_duplicates()
    df_sa = df_sa.merge(graph_info, on='graph_id', how='left')

    df_combined = pd.concat([df_g, df_sa], ignore_index=True)
    
    mode_mapping = {
        'baseline': 'SA',
        'exact_cache': 'SA-Exact Cache',
        'intelligent_memory': 'SA-LTMB'
    }
    df_combined['mode'] = df_combined['mode'].replace(mode_mapping)
    
    available_modes = df_combined['mode'].unique().tolist()
    modes_expected = ['Gurobi', 'SA', 'SA-Exact Cache', 'SA-LTMB']
    modes = [m for m in modes_expected if m in available_modes]
    
    csv_out = os.path.join(folder, "combined_results_summary.csv")
    df_combined.to_csv(csv_out, index=False)
    
    metrics = {'obj_mean': 'obj_std', 'time_s_mean': 'time_s_std', '%_covered_mean': '%_covered_std'}
    titles = {'obj_mean': 'Objective Value', 'time_s_mean': 'Runtime (s)', '%_covered_mean': 'Coverage Percentage (%)'}
    colors = ['gray', 'royalblue', 'darkorange', 'forestgreen']

    # --- 1. OVERALL COMPARISON PLOT ---
    overall = df_combined.groupby('mode').agg({
        'obj_mean': 'mean',
        'time_s_mean': 'mean',
        '%_covered_mean': 'mean'
    }).reindex(modes)
    
    overall_std = df_combined.groupby('mode').agg({
        'obj_mean': 'std', 
        'time_s_mean': 'std',
        '%_covered_mean': 'std'
    }).reindex(modes)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, (metric, std_col) in enumerate(metrics.items()):
        ax = axes[i]
        x = np.arange(len(modes))
        ax.bar(x, overall[metric], yerr=overall_std[metric], capsize=5, alpha=0.8, color=colors[:len(modes)])
        ax.set_xticks(x)
        ax.set_xticklabels(modes, rotation=0, ha='center')
        ax.set_title(f'Overall {titles[metric]}')
        ax.set_ylabel(titles[metric])
    plt.tight_layout()
    plt.savefig(os.path.join(folder, "overall_comparison.png"), dpi=300)
    plt.close()

    # --- 2. INSTANCE-WISE COMPARISON PLOT ---
    df_graph = df_combined.groupby(['graph_id', 'mode']).agg({
        'obj_mean': 'mean', 'obj_std': 'mean',
        'time_s_mean': 'mean', 'time_s_std': 'mean',
        '%_covered_mean': 'mean', '%_covered_std': 'mean'
    }).reset_index()

    graph_ids = sorted(df_graph['graph_id'].dropna().unique())
    x = np.arange(len(graph_ids))
    width = 0.8 / len(modes)

    for metric, std_col in metrics.items():
        plt.figure(figsize=(16, 6))
        sns.set_style("whitegrid")
        
        for i, mode in enumerate(modes):
            mode_data = df_graph[df_graph['mode'] == mode].set_index('graph_id').reindex(graph_ids)
            y = mode_data[metric].fillna(0).values
            yerr = mode_data[std_col].fillna(0).values
            
            offset = i * width - (len(modes) - 1) * width / 2
            plt.bar(x + offset, y, width, yerr=yerr, label=mode, capsize=3, alpha=0.8, color=colors[i])
            
        plt.xlabel('Graph ID')
        plt.ylabel(titles[metric])
        plt.title(f'{titles[metric]} Comparison Across Instances (Averaged over Intervals)')
        plt.xticks(x, [f'G{int(gid)}' for gid in graph_ids])
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(modes))
        plt.tight_layout()
        plt.savefig(os.path.join(folder, f"{metric}_instance_comparison.png"), dpi=300)
        plt.close()
        
    print("\n--- AGGREGATE RESULTS (Mean ± Std) ---")
    summary_df = pd.DataFrame()
    for col in overall.columns:
        summary_df[col + '_summary'] = overall[col].round(2).astype(str) + " ± " + overall_std[col].round(2).astype(str)
    print(summary_df)

    print("\n--- WIN COUNTS BY INSTANCE (Objective, lowest count) ---")
    inst_obj = df_combined.groupby(['graph_id', 'mode'])['obj_mean'].mean().unstack()
    print(inst_obj.idxmin(axis=1).value_counts())
    
    print("\n--- WIN COUNTS BY INSTANCE (Coverage, highest count) ---")
    inst_cov = df_combined.groupby(['graph_id', 'mode'])['%_covered_mean'].mean().unstack()
    print(inst_cov.idxmax(axis=1).value_counts())

    print("\n--- OTHER USEFUL NUMBERS ---")
    print(f"Total Graphs Evaluated: {df_combined['graph_id'].nunique()}")
    print(f"Total Records Included in Aggregation: {len(df_combined)}")

    print("\n" + "="*60)
    print("DETAILED INSTANCE-WISE RESULTS (Used for Bar Charts)")
    print("Format: Mean ± Standard Deviation")
    print("="*60)
    
    for metric, std_col in metrics.items():
        print(f"\n---> {titles[metric]} <---")
        pivot_mean = df_graph.pivot(index='graph_id', columns='mode', values=metric)
        pivot_std = df_graph.pivot(index='graph_id', columns='mode', values=std_col)
        
        combined_pivot = pd.DataFrame(index=pivot_mean.index)
        for mode in pivot_mean.columns:
            m_val = pivot_mean[mode].fillna(0).round(2).astype(str)
            s_val = pivot_std[mode].fillna(0).round(2).astype(str)
            combined_pivot[mode] = m_val + " ± " + s_val
            
        # Reorder columns to match 'modes' if possible
        present_modes = [m for m in modes if m in combined_pivot.columns]
        combined_pivot = combined_pivot[present_modes]
        print(combined_pivot.to_string())

if __name__ == "__main__":
    main()
