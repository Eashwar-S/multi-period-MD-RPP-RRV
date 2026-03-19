import os
import glob
import pandas as pd
from utils.io_utils import setup_logger, load_config

logger = setup_logger("EvalAll", "outputs/logs/evaluate_all.log")

def evaluate_all():
    config = load_config()
    
    metrics_dir = config['paths']['metrics_dir']
    
    # 1. Gather all metrics
    metric_files = glob.glob(os.path.join(metrics_dir, "*_metrics.csv"))
    if not metric_files:
        logger.error("No metrics CSV files found. Did you run the training scripts?")
        return
        
    df_list = []
    for f in metric_files:
        # Ignore summary_metrics if it exists from previous run
        if "summary_metrics" in f:
            continue
        df_list.append(pd.read_csv(f))
        
    if not df_list:
        logger.error("No valid metrics to aggregate.")
        return
        
    df_all = pd.concat(df_list, ignore_index=True)
    
    # Rank by PR-AUC (NaN = undefined = single-class test, sort to bottom)
    df_all = df_all.sort_values(by=["PR_AUC", "F1"], ascending=[False, False], na_position='last').reset_index(drop=True)
    
    logger.info("========== FINAL LEADERBOARD ==========")
    for idx, row in df_all.iterrows():
        prauc_str = f"{row['PR_AUC']:.4f}" if not pd.isna(row['PR_AUC']) else "N/A (single-class test)"
        note_val = row.get('Note', '')
        note = f"  [{note_val}]" if isinstance(note_val, str) and note_val.strip() else ""
        logger.info(f"[{idx+1}] {row['Model']} | PR-AUC: {prauc_str} | F1: {row['F1']:.4f} | "
                    f"Recall: {row['Recall']:.4f} | Precision: {row['Precision']:.4f}{note}")
        
    df_all.to_csv(os.path.join(metrics_dir, "summary_metrics.csv"), index=False)
    logger.info("Saved summary_metrics.csv")
    
    # Check if best GNN beats Logistic Regression
    try:
        gnn_candidates = [n for n in df_all['Model'].tolist() if 'GNN' in n]
        lr_row = df_all[df_all['Model'] == 'LogisticRegression'].iloc[0]
        
        for gnn_name in gnn_candidates:
            gnn_row = df_all[df_all['Model'] == gnn_name].iloc[0]
            if gnn_row['PR_AUC'] > lr_row['PR_AUC']:
                logger.info(f"[OK] {gnn_name} (PR-AUC: {gnn_row['PR_AUC']:.4f}) outperformed "
                            f"Logistic Regression (PR-AUC: {lr_row['PR_AUC']:.4f}) on PR-AUC.")
            else:
                logger.info(f"! RESULT: {gnn_name} (PR-AUC: {gnn_row['PR_AUC']:.4f}) did NOT "
                            f"outperform Logistic Regression (PR-AUC: {lr_row['PR_AUC']:.4f}) on PR-AUC.")
    except Exception as e:
        logger.warning(f"Could not compare GNN to LogisticRegression: {e}")

if __name__ == "__main__":
    evaluate_all()
