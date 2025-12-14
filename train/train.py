import numpy as np
import os
import json
import joblib
import xgboost
from sklearn.base import ClassifierMixin, BaseEstimator, clone
from catboost import CatBoostClassifier
import optuna
import warnings
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from config.config import CFG, CAT_PARAMS

class DataSampler(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, neg_pos_ratio=10.0): 
        self.estimator = estimator
        self.neg_pos_ratio = neg_pos_ratio

    def fit(self, X, y):
        X_arr = np.array(X, copy=False)
        y_arr = np.array(y, copy=False)
        
        pos_indices = np.where(y_arr == 1)[0]
        neg_indices = np.where(y_arr == 0)[0]
        
        if len(pos_indices) == 0:
            self.estimator.fit(X_arr[::10], y_arr[::10])
            self.classes_ = self.estimator.classes_
            return self

        n_neg_keep = int(len(pos_indices) * self.neg_pos_ratio)
        
        if len(neg_indices) > n_neg_keep:
            kept_neg_indices = np.random.choice(neg_indices, n_neg_keep, replace=False)
        else:
            kept_neg_indices = neg_indices
            
        final_indices = np.concatenate([pos_indices, kept_neg_indices])
        np.random.shuffle(final_indices)
        
        self.estimator.fit(X_arr[final_indices], y_arr[final_indices])
        self.classes_ = self.estimator.classes_
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(np.array(X))

    def predict(self, X):
        return self.estimator.predict(np.array(X))

def train_and_save_models(body_parts_tracked_str, switch_tr, X_tr, label, meta, section_id):
    """Train XGBoost and CatBoost models and save them to disk."""
    import catboost 
    
    # Create save directory
    save_dir = os.path.join(CFG.MODEL_SAVE_PATH, switch_tr, f"section_{section_id}")
    os.makedirs(save_dir, exist_ok=True)
    
    model_list = []
    
    for action in label.columns:
        action_mask = ~label[action].isna().values
        y_action = label[action][action_mask].values.astype(int)
        
        if not (y_action == 0).all() and len(np.unique(y_action)) >= 2:
            print(f"  Training models for action: {action}")
            
            # Train XGBoost
            model_xgb = DataSampler(clone(CFG.model), neg_pos_ratio=10.0)
            model_xgb.fit(X_tr[action_mask], y_action)
            
            # Train CatBoost
            cat_model = CatBoostClassifier(**CAT_PARAMS)
            model_cat = DataSampler(cat_model, neg_pos_ratio=10.0)
            model_cat.fit(X_tr[action_mask], y_action)
            
            # Save XGBoost model
            xgb_path = os.path.join(save_dir, f"{action}_xgboost.joblib")
            joblib.dump(model_xgb, xgb_path)
            print(f"    Saved XGBoost: {xgb_path}")
            
            cat_path = os.path.join(save_dir, f"{action}_catboost.cbm")
            model_cat.estimator.save_model(cat_path)
            joblib.dump(model_cat, os.path.join(save_dir, f"{action}_catboost_wrapper.joblib"))
            print(f"    Saved CatBoost: {cat_path}")
            
            model_list.append(action)
    
    metadata = {
        "feature_names": list(X_tr.columns),
        "actions": model_list,
        "body_parts_tracked_str": body_parts_tracked_str,
        "switch_type": switch_tr,
        "xgboost_version": xgboost.__version__,
        "catboost_version": catboost.__version__
    }
    
    metadata_path = os.path.join(save_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata: {metadata_path}")
    
    return model_list

def optimize_thresholds_optuna(y_true_dict, y_prob_dict, n_trials=100):
    """
    Optimize thresholds for multiple actions using Optuna to maximize Macro F1.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    action_names = list(y_true_dict.keys())
    
    def objective(trial):
        f1_scores = []
        for action in action_names:
            y_t = y_true_dict[action]
            y_p = y_prob_dict[action]
            
            thresh = trial.suggest_float(action, 0.1, 0.8)
            
            # Calculate F1
            score = f1_score(y_t, (y_p >= thresh).astype(int), zero_division=0)
            f1_scores.append(score)
            
        # Return F1
        return np.mean(f1_scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)
    
    best_thresholds = study.best_params
    
    print(f"Optuna Best Macro F1: {study.best_value:.4f}")
    return best_thresholds

def cross_validate_classifier(binary_classifier, X, label, meta):
    y_true_dict = {}
    y_prob_dict = {}
    
    print("  Collecting OOF predictions with StratifiedGroupKFold...")
    
    for action in label.columns:
        action_mask = ~label[action].isna().values
        X_action = X[action_mask]
        y_action = label[action][action_mask].values.astype(int)
        groups_action = meta.video_id[action_mask]
        
        # Skip if not enough data
        if len(np.unique(groups_action)) < 2 or len(np.unique(y_action)) < 2:
            continue
            
        if not (y_action == 0).all():
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                try:
                    # Use StratifiedGroupKFold
                    cv = StratifiedGroupKFold(n_splits=3)
                    
                    oof_probs = cross_val_predict(
                        binary_classifier, 
                        X_action, 
                        y_action, 
                        groups=groups_action, 
                        cv=cv, 
                        method='predict_proba'
                    )[:, 1]
                    
                    y_true_dict[action] = y_action
                    y_prob_dict[action] = oof_probs
                except Exception as e:
                    print(f"Error in CV for {action}: {e}")
                    continue

    if not y_true_dict:
        return {}

    # Optimize using Optuna
    print("  Optimizing thresholds with Optuna...")
    try:
        best_thresholds = optimize_thresholds_optuna(y_true_dict, y_prob_dict, n_trials=100)
        return best_thresholds
    except Exception as e:
        print(f"Optuna optimization failed: {e}")
        return {}

