import pandas as pd
import numpy as np
import json
from scipy.ndimage import binary_closing, binary_opening
from config.config import CFG, _OPTIMIZED_THRESHOLDS, DEFAULT_THRESHOLD, drop_body_parts
from train.preprocess import generate_mouse_data
from features.features_single import transform_single
from features.features_pair import transform_pair
from util.utility_functions import _scale, _fps_from_meta

verbose = True
submission_list = []
CURRENT_BODY_PARTS_STR = None
CURRENT_ACTION_TYPE = None

def normalize_key(body_parts_str):
    """Normalize JSON string to a canonical form for consistent lookup.
    
    Converts body_parts_str to a sorted JSON string representation
    that matches the keys in _OPTIMIZED_THRESHOLDS.
    """
    try:
        parts_list = json.loads(body_parts_str)
        # Sort and create canonical JSON string
        return json.dumps(sorted(parts_list))
    except:
        return None

def get_section_thresholds(body_parts_str, action_type=None):
    """Get the threshold dictionary for a specific section.
    
    NOTE: action_type is ignored in this version (flat structure).
    Returns a dict of {action_name: threshold} or empty dict if not found.
    """
    normalized_key = normalize_key(body_parts_str)
    if normalized_key is None:
        return {}
    
    # Try to find the section in the thresholds
    for json_key, section_data in _OPTIMIZED_THRESHOLDS.items():
        try:
            json_key_normalized = json.dumps(sorted(json.loads(json_key)))
            if json_key_normalized == normalized_key:
                return section_data  # Direct dict of {action: threshold}
        except:
            continue
    
    return {}

def predict_multiclass(pred, meta):
    """Derive multiclass predictions from a set of binary predictions.
    
    Uses per-action thresholds from hardcoded _OPTIMIZED_THRESHOLDS dict.
    Logs whether threshold was found or falls back to default.
    
    Parameters
    pred: dataframe of predicted binary probabilities, shape (n_samples, n_actions)
    meta: dataframe with columns ['video_id', 'agent_id', 'target_id', 'video_frame']
    """
    # Get section thresholds (dict of action_name -> threshold)
    # Note: FLAT structure - action_type is ignored
    section_thresholds = get_section_thresholds(CURRENT_BODY_PARTS_STR)
    
    # Gap fill: ~0.3 giây (~10 frames ở 30fps)
    # Min duration: ~0.1 giây (~3-5 frames)
    GAP_FILL_SIZE = 10
    MIN_DURATION_SIZE = 5 
    
    # 1. Apply per-action thresholds to create a mask
    binary_pred = pd.DataFrame(0, index=pred.index, columns=pred.columns)
    
    for action in pred.columns:
        # Normalize action name for lookup
        action_key = action.lower().strip()
        
        # Get threshold for this specific action
        if action_key in section_thresholds:
            thresh = section_thresholds[action_key]
            print(f'    ✓ Threshold found for action {action}: {thresh:.4f}')
        else:
            thresh = DEFAULT_THRESHOLD
            print(f'    ⚠ Threshold not found for action {action}, fall back to {DEFAULT_THRESHOLD}')
        
        binary_pred[action] = (pred[action] >= thresh).astype(int)

    # --- POST-PROCESSING ---
    for col in binary_pred.columns:
        mask = binary_pred[col].values
        
        # Gap Filling: nối các đoạn 1 bị đứt quãng bởi các số 0 ngắn
        structure_gap = np.ones(GAP_FILL_SIZE)
        mask_filled = binary_closing(mask, structure=structure_gap).astype(int)
        
        # Min Duration Filtering: xóa các đoạn 1 ngắn hơn kích thước structure
        structure_min = np.ones(MIN_DURATION_SIZE)
        mask_clean = binary_opening(mask_filled, structure=structure_min).astype(int)
        
        binary_pred[col] = mask_clean

    # 2. Mask the probabilities
    masked_pred = pred * binary_pred
    
    # 3. Find the action with the highest probability AMONG those that passed
    best_action_indices = np.argmax(masked_pred.values, axis=1)
    
    final_actions = np.full(len(pred), -1)
    
    # Only consider rows with at least one action passing threshold
    has_action_mask = binary_pred.sum(axis=1) > 0
    final_actions[has_action_mask] = best_action_indices[has_action_mask]
    
    ama = pd.Series(final_actions, index=meta.video_frame)
    # Keep only start and stop frames
    changes_mask = (ama != ama.shift(1)).values
    ama_changes = ama[changes_mask]
    meta_changes = meta[changes_mask]
    # mask selects the start frames
    mask = ama_changes.values >= 0
    mask[-1] = False  
    submission_part = pd.DataFrame({
        'video_id': meta_changes['video_id'][mask].values,
        'agent_id': meta_changes['agent_id'][mask].values,
        'target_id': meta_changes['target_id'][mask].values,
        'action': pred.columns[ama_changes[mask].values],
        'start_frame': ama_changes.index[mask],
        'stop_frame': ama_changes.index[1:][mask[:-1]]
    })
    
    # Handle action extending to end of video
    stop_video_id = meta_changes['video_id'][1:][mask[:-1]].values
    stop_agent_id = meta_changes['agent_id'][1:][mask[:-1]].values
    stop_target_id = meta_changes['target_id'][1:][mask[:-1]].values
    for i in range(len(submission_part)):
        video_id = submission_part.video_id.iloc[i]
        agent_id = submission_part.agent_id.iloc[i]
        target_id = submission_part.target_id.iloc[i]
        
        if (video_id != stop_video_id[i]) or (agent_id != stop_agent_id[i]) or (target_id != stop_target_id[i]):
            submission_part.stop_frame.iloc[i] = meta.video_frame.iloc[-1] + 1
            
    return submission_part

def submit_with_loaded_models(body_parts_tracked_str, switch_tr, model_list, metadata, section_id):
    "Produce predictions using pre-loaded models."
    global CURRENT_BODY_PARTS_STR, CURRENT_ACTION_TYPE
    CURRENT_BODY_PARTS_STR = body_parts_tracked_str
    CURRENT_ACTION_TYPE = switch_tr
    body_parts_tracked = json.loads(body_parts_tracked_str)
    if len(body_parts_tracked) > 5:
        body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    
    test = pd.read_csv(CFG.test_path)
    test_subset = test[test.body_parts_tracked == body_parts_tracked_str]
    generator = generate_mouse_data(test_subset, "test",
                                    generate_single=(switch_tr == "single"),
                                    generate_pair=(switch_tr == "pair"))
    
    fps_lookup = (
        test_subset[["video_id", "frames_per_second"]]
        .drop_duplicates("video_id")
        .set_index("video_id")["frames_per_second"]
        .to_dict()
    )
    
    for switch_te, data_te, meta_te, actions_te in generator:
        assert switch_te == switch_tr
        try:
            fps_i = _fps_from_meta(meta_te, fps_lookup)
            
            if switch_te == "single":
                X_te = transform_single(data_te, body_parts_tracked, fps_i)
            else:
                X_te = transform_pair(data_te, body_parts_tracked, fps_i)
            
            if verbose and len(X_te) == 0:
                print("ERROR: X_te is empty")
            del data_te
            
            # Reindex features
            expected_features = metadata.get("feature_names", [])
            if not expected_features:
                 pass 
            else:
                for feat in expected_features:
                    if feat not in X_te.columns:
                        X_te[feat] = 0
                X_te = X_te[expected_features]
            
            # Compute predictions
            pred = pd.DataFrame(index=meta_te.video_frame)
            for action, model_xgb, model_cat in model_list:
                if action in actions_te:
                    try:
                        p_xgb = model_xgb.predict_proba(X_te)[:, 1]
                        p_cat = model_cat.predict_proba(X_te)[:, 1]
                        pred[action] = (p_xgb + p_cat) / 2.0
                    except Exception as e:
                        print(f"Error predicting {action}: {e}")
            del X_te
            
            # Probability Smoothing
            ws = _scale(15, fps_i)
            pred = pred.rolling(window=ws, min_periods=1, center=True).mean()
            
            if pred.shape[1] != 0:
                submission_part = predict_multiclass(pred, meta_te)
                submission_list.append(submission_part)
            else:
                if verbose:
                    print("  ERROR: no useful training data")
        except KeyError:
            if verbose:
                print(f"  ERROR: KeyError because of missing bodypart ({switch_tr})")
            del data_te
            
    return submission_list
