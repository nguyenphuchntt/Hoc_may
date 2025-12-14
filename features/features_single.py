import pandas as pd
import numpy as np
import itertools
from util.utility_functions import smooth_coordinates, _scale, safe_sqrt, _scale_signed
from features.features_common import (
    add_curvature_features,
    add_multiscale_features,
    add_state_features,
    add_longrange_features,
    add_advanced_kinematics,
    add_fft_features,
    add_body_angle_features,
    add_egocentric_features,
    add_grooming_features,
    add_temporal_asymmetry
)

def transform_single(single_mouse, body_parts_tracked, fps):
    """Transform from cartesian coordinates to distance representation.

    Parameters:
    single_mouse: dataframe with coordinates of the body parts of one mouse
                  shape (n_samples, n_body_parts * 2)
                  two-level MultiIndex on columns
    body_parts_tracked: list of body parts
    """
    available_body_parts = single_mouse.columns.get_level_values(0)
    
    # Smooth coordinates to reduce noise
    single_mouse = smooth_coordinates(single_mouse, sigma=1.5)
    
    # X là toàn bộ khoảng cách giữa các bộ phận của con chuột
    X = pd.DataFrame({
            f"{part1}+{part2}": np.square(single_mouse[part1] - single_mouse[part2]).sum(axis=1, skipna=False)
            for part1, part2 in itertools.combinations(body_parts_tracked, 2) if part1 in available_body_parts and part2 in available_body_parts
        })
    X = X.reindex(columns=[f"{part1}+{part2}" for part1, part2 in itertools.combinations(body_parts_tracked, 2)], copy=False)

    if all(p in single_mouse.columns for p in ['ear_left', 'ear_right', 'tail_base']):
        # lag ~ 10 frame trong 30fps
        lag = _scale(10, fps) 
        shifted = single_mouse[['ear_left', 'ear_right', 'tail_base']].shift(lag) 
        X = pd.concat([
            X, 
            pd.DataFrame({
                'speed_left': np.square(single_mouse['ear_left'] - shifted['ear_left']).sum(axis=1, skipna=False),
                'speed_right': np.square(single_mouse['ear_right'] - shifted['ear_right']).sum(axis=1, skipna=False),
                'speed_left2': np.square(single_mouse['ear_left'] - shifted['tail_base']).sum(axis=1, skipna=False),
                'speed_right2': np.square(single_mouse['ear_right'] - shifted['tail_base']).sum(axis=1, skipna=False),
            })
        ], axis=1)

    new_features = {}
    # Elongation: độ kéo dài của cơ thể
    if 'nose+tail_base' in X.columns and "ear_left+ear_right" in X.columns:
        new_features['elong'] = safe_sqrt(X['nose+tail_base'] / (X['ear_left+ear_right'] + 1e-6))

    # Góc từ mũi -> thân -> đuôi
    if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
        v1 = single_mouse['nose']-single_mouse['body_center']
        v2 = single_mouse['tail_base'] - single_mouse['body_center']
        new_features['body_ang'] = (v1['x'] * v2['x'] + v1['y'] * v2['y']) / (safe_sqrt(v1['x']**2 + v1['y']**2) * safe_sqrt(v2['x']**2 + v2['y']**2))

    # Khoảng cách giữa mũi và đuôi
    if all(p in available_body_parts for p in ['nose', 'tail_base']):
        nt_dist = safe_sqrt((single_mouse['nose']['x'] - single_mouse['tail_base']['x'])**2 +
                          (single_mouse['nose']['y'] - single_mouse['tail_base']['y'])**2)
        for lag in [10, 20, 40]:
            l = _scale(lag, fps)
            new_features[f'nt_lg{lag}'] = nt_dist.shift(l) # khoảng cách giữa nose-tail trong quá khứ
            new_features[f'nt_df{lag}'] = nt_dist - nt_dist.shift(l) # Độ thay đổi so với hiện tại

    # Rolling statistic dựa trên body center
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']

        for w in [5, 15, 30, 60]:
            w_scale = _scale(w, fps)
            roll = dict(window=w_scale, min_periods=1, center=True)
            new_features[f'cx_mean_{w}'] = center_x.rolling(**roll).mean()
            new_features[f'cy_mean_{w}'] = center_y.rolling(**roll).mean()
            new_features[f'cx_std_{w}'] = center_x.rolling(**roll).var().clip(lower=0).pow(0.5)
            new_features[f'cy_std_{w}'] = center_y.rolling(**roll).var().clip(lower=0).pow(0.5)
            new_features[f'cx_range_{w}'] = center_x.rolling(**roll).max() - center_x.rolling(**roll).min()
            new_features[f'cy_range_{w}'] = center_y.rolling(**roll).max() - center_y.rolling(**roll).min()
            new_features[f'variablitiy_{w}'] = safe_sqrt(center_x.diff().rolling(w_scale, min_periods=1).var().clip(lower=0) + 
                                             center_y.diff().rolling(w_scale, min_periods=1).var().clip(lower=0))
            new_features[f'displacement_{w}'] = safe_sqrt(center_x.diff().rolling(w_scale, min_periods=1).sum()**2 + 
                                              center_y.diff().rolling(w_scale, min_periods=1).sum()**2)
            
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    # Call helper functions (they now use pd.concat internally)
    if 'body_center' in available_body_parts:
        center_x = single_mouse['body_center']['x']
        center_y = single_mouse['body_center']['y']
        
        X = add_curvature_features(X, center_x, center_y, fps)
        X = add_multiscale_features(X, center_x, center_y, fps)
        X = add_state_features(X, center_x, center_y, fps)
        X = add_longrange_features(X, center_x, center_y, fps)
        
        # New features: Advanced kinematics
        X = add_advanced_kinematics(X, center_x, center_y, fps)
        
        # New features: FFT on speed signal
        speed_signal = safe_sqrt(center_x.diff()**2 + center_y.diff()**2)
        X = add_fft_features(X, speed_signal, 'speed', fps, window_size=120)
        
        # New features: Body angles (if body parts available)
        if all(p in available_body_parts for p in ['nose', 'body_center', 'tail_base']):
            X = add_body_angle_features(X, 
                                       single_mouse['nose']['x'], single_mouse['nose']['y'],
                                       center_x, center_y,
                                       single_mouse['tail_base']['x'], single_mouse['tail_base']['y'],
                                       fps)

    if all(p in available_body_parts for p in ['ear_left', 'ear_right']):
        ear_dist = safe_sqrt((single_mouse['ear_left']['x'] - single_mouse['ear_right']['x'])**2 + 
                           (single_mouse['ear_left']['y'] - single_mouse['ear_right']['y'])**2)
        
        new_features_ear = {}
        for offset in [-30, -20, -10, 10, 20, 30]:
            o = _scale_signed(offset, fps)
            new_features_ear[f'ear_dist_o{offset}'] = ear_dist.shift(-o)
        
        w = _scale(30, fps)
        new_features_ear['ear_consistency'] = ear_dist.rolling(w, min_periods=1, center=True).var().clip(lower=0).pow(0.5) / (ear_dist.rolling(w, min_periods=1, center=True).mean() + 1e-6)
        
        if new_features_ear:
            X = pd.concat([X, pd.DataFrame(new_features_ear, index=X.index)], axis=1)

    # --- ĐOẠN CODE THÊM MỚI ---
    # 1. Thêm Egocentric (Quan trọng nhất)
    X = add_egocentric_features(X, single_mouse, fps)
    
    # 2. Thêm Grooming Features (Cải thiện lớp 'Grooming' và 'Other')
    X = add_grooming_features(X, single_mouse, fps)
    
    # 3. Thêm Temporal Asymmetry (Cải thiện lớp 'Attack' và 'Chase')
    if 'body_center' in available_body_parts:
        cx = single_mouse['body_center']['x']
        cy = single_mouse['body_center']['y']
        X = add_temporal_asymmetry(X, cx, cy, fps)
    # --------------------------
    return X.astype(np.float32, copy=False)
