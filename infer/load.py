import os
import glob
import joblib
import json
from config.config import CFG

verbose = True

def load_models(switch_type, section_id):    
    load_dir = os.path.join(CFG.MODEL_LOAD_PATH, switch_type, f"section_{section_id}")

    if not os.path.exists(load_dir):
        print(f"ERROR: Model directory not found: {load_dir}")
        return [], {}

    metadata = {}
    actions = []
    
    # Check for metadata.json
    metadata_path = os.path.join(load_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        actions = metadata.get("actions", [])
        if not actions and verbose: print("  Warning: 'actions' list empty in metadata")
    else:
        print(f"Notice: metadata.json not found in {load_dir}. Inferring actions from filenames.")
        xgb_files = glob.glob(os.path.join(load_dir, "*_xgboost.joblib"))
        for fpath in xgb_files:
            fname = os.path.basename(fpath)
            action = fname.replace("_xgboost.joblib", "")
            actions.append(action)
        actions.sort()
        metadata["actions"] = actions
        metadata["feature_names"] = [] 

    model_list = []
    for action in actions:
        try:
            # Load XGBoost
            xgb_path = os.path.join(load_dir, f"{action}_xgboost.joblib")
            if not os.path.exists(xgb_path):
                 print(f"Skipping {action}: XGBoost model not found at {xgb_path}")
                 continue
                 
            model_xgb = joblib.load(xgb_path)
            
            # Load CatBoost wrapper
            cat_wrapper_path = os.path.join(load_dir, f"{action}_catboost_wrapper.joblib")
            if os.path.exists(cat_wrapper_path):
                model_cat = joblib.load(cat_wrapper_path)
            else:
                 print(f"Skipping {action}: CatBoost wrapper not found at {cat_wrapper_path}")
                 continue
            
            model_list.append((action, model_xgb, model_cat))
            print(f"Loaded: {action}")
        except Exception as e:
            print(f"Error loading {action}: {e}")
    
    return model_list, metadata
