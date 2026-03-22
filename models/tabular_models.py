from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def get_tabular_models(random_state=42):
    """
    Returns a dictionary of un-initialized or default-initialized models.
    """
    models = {
        "LogisticRegression": LogisticRegression(random_state=random_state, class_weight='balanced', max_iter=1000),
        "RandomForest": RandomForestClassifier(random_state=random_state, class_weight='balanced', n_estimators=100, max_depth=7, min_samples_split=10, min_samples_leaf=5),
        "SVM": SVC(random_state=random_state, class_weight='balanced', probability=True),
        "MLP": MLPClassifier(random_state=random_state, hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True)
    }
    
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            random_state=random_state, 
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5.0, # Will be overridden by actual class ratio if known
            eval_metric='logloss'
        )
        
    return models
