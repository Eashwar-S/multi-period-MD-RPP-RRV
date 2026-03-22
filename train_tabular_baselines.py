import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from utils.io_utils import setup_logger, load_config
from utils.metrics_utils import compute_metrics, tune_threshold_on_validation, plot_pr_curve, plot_confusion_matrix
from models.tabular_models import get_tabular_models
from utils.reproducibility import set_seed
import pickle

logger = setup_logger("TrainTabular", "outputs/logs/train_tabular.log")

TRAIN_AREAS = ['Arlington, Texaas, US', 'Boston, Massachusetts US', 'College Park, Maryland, US', 'Indianapolis, US']
TEST_AREAS = [
    'Little Rock, Arkansas, US', 'Louisville, Kentucky, US', 'Lubbock, Texaas, US', 
    'Memphis, US', 'Nashville, Tennessee, US', 'newyork, US', 'Oklahoma, US', 'Philadelphia, US'
]

def train_and_eval_tabular():
    config = load_config()
    set_seed(config['project']['seed'])
    
    data_path = os.path.join(config['paths']['processed_data_dir'], "processed_tabular.csv")
    df = pd.read_csv(data_path)
    
    with open(os.path.join(config['paths']['processed_data_dir'], "schema_report.json"), "r") as f:
        schema = json.load(f)
        
    features = schema['all_model_features']
    target = schema['target_col']
    
    # In tabular models, we can either predict edge or node. 
    # Since the tabular df is node-level, we'll train these models on node-level binary classification.
    # Note: If edge-level is strictly required, we'd need to expand to pairs.
    # The instructions allow: "If the dataset is node-level, keep the model node-level".
    
    # Drop NaNs
    df = df.dropna(subset=features + [target])
    df['date'] = pd.to_datetime(df['date'], utc=True)
    
    df['split'] = 'unknown'
    for area in TRAIN_AREAS:
        area_mask = df['area_id'] == area
        dates = sorted(df.loc[area_mask, 'date'].unique())
        if len(dates) == 0: continue
        split_idx = max(1, int(len(dates) * 0.8))
        train_dates = dates[:split_idx]
        val_dates = dates[split_idx:]
        if len(val_dates) == 0: val_dates = train_dates
        df.loc[area_mask & df['date'].isin(train_dates), 'split'] = 'train'
        df.loc[area_mask & df['date'].isin(val_dates), 'split'] = 'val'
        
    for area in TEST_AREAS:
        df.loc[df['area_id'] == area, 'split'] = 'test'
    
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    
    X_train = df_train[features].values
    y_train = df_train[target].values
    
    X_val = df_val[features].values
    y_val = df_val[target].values
    
    logger.info(f"Train/Val sizes (combined over {len(TRAIN_AREAS)} graphs): {len(X_train)} / {len(X_val)}")
    
    models = get_tabular_models(config['project']['seed'])
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_test_metrics = []
    
    # Create outputs layout
    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    os.makedirs(config['paths']['plots_dir'], exist_ok=True)
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['predictions_dir'], exist_ok=True)
    
    df_test_all = df[(df['area_id'].isin(TEST_AREAS))]
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        with open(os.path.join(config['paths']['models_dir'], f"{name}.pkl"), 'wb') as f:
            pickle.dump(model, f)
            
        # Get probabilities for Train and Val
        if hasattr(model, "predict_proba"):
            probs_train = model.predict_proba(X_train)[:, 1]
            probs_val = model.predict_proba(X_val)[:, 1]
        else:
            probs_train = model.decision_function(X_train)
            probs_val = model.decision_function(X_val)
            
            p_min, p_max = probs_train.min(), probs_train.max()
            probs_train = (probs_train - p_min) / (p_max - p_min + 1e-8)
            probs_val = (probs_val - probs_val.min()) / (probs_val.max() - probs_val.min() + 1e-8)
            
        # Tune threshold
        logger.info(f"Tuning threshold on validation set for {name}...")
        best_threshold = tune_threshold_on_validation(y_val, probs_val, metric="F1")
        
        train_res = compute_metrics(y_train, probs_train, threshold=best_threshold)
        val_res = compute_metrics(y_val, probs_val, threshold=best_threshold)
        
        logger.info(f"[{name}] Best threshold: {best_threshold:.3f} | TRAIN PR-AUC: {train_res['PR_AUC']:.4f} | VAL PR-AUC: {val_res['PR_AUC']:.4f}")
        
        # Test Evaluation on 8 Graphs
        for area in TEST_AREAS:
            df_a_test = df_test_all[df_test_all['area_id'] == area]
            if len(df_a_test) == 0:
                continue
                
            X_te = df_a_test[features].values
            y_te = df_a_test[target].values
            
            if hasattr(model, "predict_proba"):
                probs_test = model.predict_proba(X_te)[:, 1]
            else:
                probs_test = model.decision_function(X_te)
                probs_test = (probs_test - probs_test.min()) / (probs_test.max() - probs_test.min() + 1e-8)
                
            test_metrics = compute_metrics(y_te, probs_test, threshold=best_threshold)
            test_metrics["Model"] = name
            test_metrics["Area"] = area
            test_metrics["Threshold"] = best_threshold
            all_test_metrics.append(test_metrics)
            
    if len(all_test_metrics) > 0:
        metrics_df = pd.DataFrame(all_test_metrics)
        metrics_df.to_excel(os.path.join(config['paths']['metrics_dir'], f"tabular_test_metrics_{timestamp}.xlsx"), index=False)
        
        for name in models.keys():
            if 'Model' in metrics_df.columns:
                m_df = metrics_df[metrics_df['Model'] == name]
                if len(m_df) > 0:
                    avg_prauc = m_df['PR_AUC'].mean()
                    avg_f1 = m_df['F1'].mean()
                    logger.info(f"{name} Test Average PR-AUC: {avg_prauc:.4f}, Average F1: {avg_f1:.4f}")
    else:
        logger.warning("No test metrics were computed (TEST_AREAS might be empty or all NaNs).")
            
    logger.info("Tabular baseline training and evaluation complete.")

if __name__ == "__main__":
    train_and_eval_tabular()
