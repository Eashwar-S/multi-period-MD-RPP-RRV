import os
import json
import numpy as np
import pandas as pd
from utils.io_utils import setup_logger, load_config
from utils.metrics_utils import compute_metrics, tune_threshold_on_validation, plot_pr_curve, plot_confusion_matrix
from models.tabular_models import get_tabular_models
from utils.reproducibility import set_seed
import pickle

logger = setup_logger("TrainTabular", "outputs/logs/train_tabular.log")

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
    
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    df_test = df[df['split'] == 'test']
    
    X_train = df_train[features].values
    y_train = df_train[target].values
    
    X_val = df_val[features].values
    y_val = df_val[target].values
    
    X_test = df_test[features].values
    y_test = df_test[target].values
    
    logger.info(f"Train/Val/Test sizes: {len(X_train)} / {len(X_val)} / {len(X_test)}")
    
    models = get_tabular_models(config['project']['seed'])
    
    all_test_metrics = []
    
    # Create outputs layout
    os.makedirs(config['paths']['metrics_dir'], exist_ok=True)
    os.makedirs(config['paths']['plots_dir'], exist_ok=True)
    os.makedirs(config['paths']['models_dir'], exist_ok=True)
    os.makedirs(config['paths']['predictions_dir'], exist_ok=True)
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Save model
        with open(os.path.join(config['paths']['models_dir'], f"{name}.pkl"), 'wb') as f:
            pickle.dump(model, f)
            
        # Get probabilities
        if hasattr(model, "predict_proba"):
            probs_val = model.predict_proba(X_val)[:, 1]
            probs_test = model.predict_proba(X_test)[:, 1]
        else:
            # Fallback for models without proba like simple SVM without probability=True
            probs_val = model.decision_function(X_val)
            probs_test = model.decision_function(X_test)
            # Normalize to 0-1
            probs_val = (probs_val - probs_val.min()) / (probs_val.max() - probs_val.min() + 1e-8)
            probs_test = (probs_test - probs_test.min()) / (probs_test.max() - probs_test.min() + 1e-8)
            
        logger.info(f"Tuning threshold on validation set for {name}...")
        best_threshold = tune_threshold_on_validation(y_val, probs_val, metric="F1")
        logger.info(f"Best threshold for {name}: {best_threshold:.3f}")
        
        test_metrics = compute_metrics(y_test, probs_test, threshold=best_threshold)
        test_metrics["Model"] = name
        test_metrics["Threshold"] = best_threshold
        all_test_metrics.append(test_metrics)
        
        logger.info(f"{name} Test PR-AUC: {test_metrics['PR_AUC']:.4f}, F1: {test_metrics['F1']:.4f}")
        
        # Save plots
        plot_pr_curve(y_test, probs_test, name, os.path.join(config['paths']['plots_dir'], f"pr_curve_{name}.png"))
        y_pred_test = (probs_test >= best_threshold).astype(int)
        plot_confusion_matrix(y_test, y_pred_test, name, os.path.join(config['paths']['plots_dir'], f"confusion_matrix_{name}.png"))
        
        # Save predictions
        df_test_preds = df_test[['date', 'node_id', target]].copy()
        df_test_preds['prob'] = probs_test
        df_test_preds['pred'] = y_pred_test
        df_test_preds.to_csv(os.path.join(config['paths']['predictions_dir'], f"{name}_test_predictions.csv"), index=False)
        
    metrics_df = pd.DataFrame(all_test_metrics)
    metrics_df.to_csv(os.path.join(config['paths']['metrics_dir'], "tabular_metrics.csv"), index=False)
    logger.info("Tabular baseline training complete.")

if __name__ == "__main__":
    train_and_eval_tabular()
