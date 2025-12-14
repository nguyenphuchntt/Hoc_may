import pandas as pd
import numpy as np
import itertools
from util.utility_functions import smooth_coordinates, _scale, safe_sqrt, _scale_signed

def add_interaction_features(X, mouse_pair, avail_A, avail_B, fps):
    """Social interaction features (windows scaled by fps)."""
    if 'body_center' not in avail_A or 'body_center' not in avail_B:
        return X

    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    rel_dist = safe_sqrt(rel_x**2 + rel_y**2)

    # per-frame velocities (cm/frame)
    A_vx = mouse_pair['A']['body_center']['x'].diff()
    A_vy = mouse_pair['A']['body_center']['y'].diff()
    B_vx = mouse_pair['B']['body_center']['x'].diff()
    B_vy = mouse_pair['B']['body_center']['y'].diff()

    A_lead = (A_vx * rel_x + A_vy * rel_y) / (safe_sqrt(A_vx**2 + A_vy**2) * rel_dist + 1e-6)
    B_lead = (B_vx * (-rel_x) + B_vy * (-rel_y)) / (safe_sqrt(B_vx**2 + B_vy**2) * rel_dist + 1e-6)

    new_features = {}
    for window in [30, 60]:
        ws = _scale(window, fps)
        new_features[f'A_ld{window}'] = A_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()
        new_features[f'B_ld{window}'] = B_lead.rolling(ws, min_periods=max(1, ws // 6)).mean()

    approach = -rel_dist.diff()  # decreasing distance => positive approach
    chase = approach * B_lead
    w = 30
    ws = _scale(w, fps)
    new_features[f'chase_{w}'] = chase.rolling(ws, min_periods=max(1, ws // 6)).mean()

    for window in [60, 120]:
        ws = _scale(window, fps)
        A_sp = safe_sqrt(A_vx**2 + A_vy**2)
        B_sp = safe_sqrt(B_vx**2 + B_vy**2)
        new_features[f'sp_cor{window}'] = A_sp.rolling(ws, min_periods=max(1, ws // 6)).corr(B_sp)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_enhanced_social_features(X, mouse_pair, avail_A, avail_B, fps):
    """Add approach angle and facing target features."""
    if not all(p in avail_A for p in ['nose', 'tail_base', 'body_center']):
        return X
    if not all(p in avail_B for p in ['nose', 'tail_base', 'body_center']):
        return X
    
    # Relative position
    rel_x = mouse_pair['A']['body_center']['x'] - mouse_pair['B']['body_center']['x']
    rel_y = mouse_pair['A']['body_center']['y'] - mouse_pair['B']['body_center']['y']
    
    # A's heading direction
    A_head_x = mouse_pair['A']['nose']['x'] - mouse_pair['A']['tail_base']['x']
    A_head_y = mouse_pair['A']['nose']['y'] - mouse_pair['A']['tail_base']['y']
    A_heading = np.arctan2(A_head_y, A_head_x)
    
    # Angle from A to B
    angle_to_B = np.arctan2(rel_y, rel_x)
    
    # Approach angle: difference between heading and direction to other mouse
    approach_angle = A_heading - angle_to_B
    # Normalize to [-pi, pi]
    approach_angle = approach_angle.replace([np.inf, -np.inf], np.nan)
    approach_angle = np.arctan2(np.sin(approach_angle), np.cos(approach_angle))
    
    new_features = {}
    new_features['approach_angle'] = approach_angle
    
    # Binary facing indicator (facing if angle < 45 degrees)
    new_features['facing_target'] = (np.abs(approach_angle.fillna(np.pi)) < np.pi/4).astype(float)    
    # Continuous facing feature (Cosine)
    new_features['facing_cosine'] = np.cos(approach_angle)
    
    # Approach speed (Projected Velocity of A towards B)
    # 1. Vector from A to B
    vec_AB_x = mouse_pair['B']['body_center']['x'] - mouse_pair['A']['body_center']['x']
    vec_AB_y = mouse_pair['B']['body_center']['y'] - mouse_pair['A']['body_center']['y']
    dist_AB = safe_sqrt(vec_AB_x**2 + vec_AB_y**2) + 1e-6

    # 2. Velocity of A
    vel_A_x = mouse_pair['A']['body_center']['x'].diff().fillna(0)
    vel_A_y = mouse_pair['A']['body_center']['y'].diff().fillna(0)

    # 3. Projected velocity
    new_features['approach_speed'] = (vel_A_x * vec_AB_x + vel_A_y * vec_AB_y) / dist_AB * fps

    
    # Rolling statistics
    ws = _scale(30, fps)
    new_features['facing_pct_30'] = new_features['facing_target'].rolling(ws, min_periods=max(1, ws//6)).mean()
    
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def transform_pair(mouse_pair, body_parts_tracked, fps):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    mouse_pair: dataframe with coordinates of the body parts of two mice
                  shape (n_samples, 2 * n_body_parts * 2)
                  three-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    # drop_body_parts =  ['ear_left', 'ear_right',
    #                     'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    #                     'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    #                     'tail_midpoint']
    # if len(body_parts_tracked) > 5:
    #     body_parts_tracked = [b for b in body_parts_tracked if b not in drop_body_parts]
    available_body_parts_A = mouse_pair['A'].columns.get_level_values(0)
    available_body_parts_B = mouse_pair['B'].columns.get_level_values(0)
    
    # Smooth coordinates to reduce noise
    mouse_pair['A'] = smooth_coordinates(mouse_pair['A'], sigma=1.5)
    mouse_pair['B'] = smooth_coordinates(mouse_pair['B'], sigma=1.5)
    
    ETHOLOGICAL_PAIRS = [
        ('nose', 'nose'),
        ('nose', 'tail_base'),
        ('tail_base', 'nose'),
        ('nose', 'body_center'),
        ('body_center', 'nose'),
        ('body_center', 'body_center'),
        ('tail_base', 'tail_base')
    ]
    
    X = pd.DataFrame({
            f"12+{part1}+{part2}": np.square(mouse_pair['A'][part1] - mouse_pair['B'][part2]).sum(axis=1, skipna=False)
            for part1, part2 in ETHOLOGICAL_PAIRS if part1 in available_body_parts_A and part2 in available_body_parts_B
        })
    X = X.reindex(columns=[f"12+{part1}+{part2}" for part1, part2 in ETHOLOGICAL_PAIRS], copy=False)

    if ('A', 'ear_left') in mouse_pair.columns and ('B', 'ear_left') in mouse_pair.columns:
        lag = _scale(10, fps)
        shifted_A = mouse_pair['A']['ear_left'].shift(lag)
        shifted_B = mouse_pair['B']['ear_left'].shift(lag)
        X = pd.concat([
            X,
            pd.DataFrame({
                'speed_left_A': np.square(mouse_pair['A']['ear_left'] - shifted_A).sum(axis=1, skipna=False),
                'speed_left_AB': np.square(mouse_pair['A']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
                'speed_left_B': np.square(mouse_pair['B']['ear_left'] - shifted_B).sum(axis=1, skipna=False),
            })
        ], axis=1)

    new_features = {}
    # góc giữa 2 con chuột
    if all(p in available_body_parts_A for p in ['nose', 'tail_base']) and all(p in available_body_parts_B for p in ['nose', 'tail_base']):
        dir_A = mouse_pair['A']['nose'] - mouse_pair['A']['tail_base']
        dir_B = mouse_pair['B']['nose'] - mouse_pair['B']['tail_base']
        new_features['rel_ori'] = (dir_A['x'] * dir_B['x'] + dir_A['y'] * dir_B['y']) / (
            safe_sqrt(dir_A['x']**2 + dir_A['y']**2) * safe_sqrt(dir_B['x']**2 + dir_B['y']**2) + 1e-6)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # Add interaction features
    X = add_interaction_features(X, mouse_pair, available_body_parts_A, available_body_parts_B, fps)
    
    # Add enhanced social features
    X = add_enhanced_social_features(X, mouse_pair, available_body_parts_A, available_body_parts_B, fps)

    return X.astype(np.float32, copy=False)
