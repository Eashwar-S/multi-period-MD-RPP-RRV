import os
import yaml
import pandas as pd
from utils.io_utils import setup_logger, load_config

logger = setup_logger("ExportResults", "outputs/logs/export_results.log")

def export_results():
    config = load_config()
    metrics_dir = config['paths']['metrics_dir']
    
    summary_csv = os.path.join(metrics_dir, "summary_metrics.csv")
    if not os.path.exists(summary_csv):
        logger.error("summary_metrics.csv not found! Run evaluate_all.py first.")
        return
        
    df_summary = pd.read_csv(summary_csv)
    
    # Create an Excel Writer
    out_xlsx = os.path.join(metrics_dir, "summary_metrics.xlsx")
    with pd.ExcelWriter(out_xlsx, engine='xlsxwriter') as writer:
        
        # 1. Overall Summary
        df_summary.to_excel(writer, sheet_name='overall_summary', index=False)
        worksheet = writer.sheets['overall_summary']
        worksheet.autofit()
        
        # 2. Config Used
        # Convert config dict to flat dataframe for easy viewing in excel
        config_rows = []
        for k1, v1 in config.items():
            if isinstance(v1, dict):
                for k2, v2 in v1.items():
                    config_rows.append({"Category": k1, "Key": k2, "Value": str(v2)})
            else:
                config_rows.append({"Category": "root", "Key": k1, "Value": str(v1)})
                
        df_config = pd.DataFrame(config_rows)
        df_config.to_excel(writer, sheet_name='config_used', index=False)
        worksheet = writer.sheets['config_used']
        worksheet.autofit()
        
        # Note: If per-date metrics or validation summaries were serialized in training loop,
        # they would be read and added here safely. For the scope of this file, we create default tabs.
        df_blank = pd.DataFrame(["Not recorded in this run"], columns=["Notes"])
        
        df_blank.to_excel(writer, sheet_name='validation_summary', index=False)
        df_blank.to_excel(writer, sheet_name='test_summary', index=False)
        df_blank.to_excel(writer, sheet_name='threshold_sweeps', index=False)
        df_blank.to_excel(writer, sheet_name='per_date_metrics', index=False)
        
    logger.info(f"Exported multi-sheet workbook to {out_xlsx}")

if __name__ == "__main__":
    export_results()
