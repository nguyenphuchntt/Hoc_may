import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, spectrogram
from util.utility_functions import safe_sqrt, _scale, _scale_signed

def add_curvature_features(X, center_x, center_y, fps):
    """Trajectory curvature (window lengths scaled by fps)."""
    vel_x = center_x.diff()
    vel_y = center_y.diff()
    acc_x = vel_x.diff()
    acc_y = vel_y.diff()

    cross_prod = vel_x * acc_y - vel_y * acc_x
    vel_mag = safe_sqrt(vel_x**2 + vel_y**2)
    curvature = np.abs(cross_prod) / (vel_mag**3 + 1e-6)  # invariant to time scaling

    new_features = {}
    for w in [30, 60]:
        ws = _scale(w, fps)
        new_features[f'curv_mean_{w}'] = curvature.rolling(ws, min_periods=max(1, ws // 6)).mean()

    angle = np.arctan2(vel_y, vel_x)
    angle_change = np.abs(angle.diff())
    w = 30
    ws = _scale(w, fps)
    new_features[f'turn_rate_{w}'] = angle_change.rolling(ws, min_periods=max(1, ws // 6)).sum()

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_multiscale_features(X, center_x, center_y, fps):
    """Multi-scale temporal features (speed in cm/s; windows scaled by fps)."""
    # displacement per frame is already in cm (pix normalized earlier); convert to cm/s
    speed = safe_sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)
    
    # Smooth speed signal using Savitzky-Golay filter (window 15 frames)
    try:
        ws = min(15, len(speed) // 2)
        if ws >= 5 and ws % 2 == 1:  # Must be odd
            speed_smooth = pd.Series(savgol_filter(speed.fillna(0).values, ws, 3), index=speed.index)
            X['speed_smooth'] = speed_smooth
    except:
        pass
    

    new_features = {}
    scales = [10, 40, 160]
    for scale in scales:
        ws = _scale(scale, fps)
        if len(speed) >= ws:
            new_features[f'sp_m{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).mean()
            new_features[f'sp_s{scale}'] = speed.rolling(ws, min_periods=max(1, ws // 4)).var().clip(lower=0).pow(0.5)

    if len(scales) >= 2 and f'sp_m{scales[0]}' in new_features and f'sp_m{scales[-1]}' in new_features:
        new_features['sp_ratio'] = new_features[f'sp_m{scales[0]}'] / (new_features[f'sp_m{scales[-1]}'] + 1e-6)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_state_features(X, center_x, center_y, fps):
    """Behavioral state transitions; bins adjusted so semantics are fps-invariant."""
    speed = safe_sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    w_ma = _scale(15, fps)
    speed_ma = speed.rolling(w_ma, min_periods=max(1, w_ma // 3)).mean()

    try:
        # Original bins (cm/frame): [-inf, 0.5, 2.0, 5.0, inf]
        # Convert to cm/s by multiplying by fps to keep thresholds consistent across fps.
        bins = [-np.inf, 0.5 * fps, 2.0 * fps, 5.0 * fps, np.inf]
        speed_states = pd.cut(speed_ma, bins=bins, labels=[0, 1, 2, 3]).astype(float)

        new_features = {}
        for window in [60, 120]:
            ws = _scale(window, fps)
            if len(speed_states) >= ws:
                for state in [0, 1, 2, 3]:
                    new_features[f's{state}_{window}'] = (
                        (speed_states == state).astype(float)
                        .rolling(ws, min_periods=max(1, ws // 6)).mean()
                    )
                state_changes = (speed_states != speed_states.shift(1)).astype(float)
                new_features[f'trans_{window}'] = state_changes.rolling(ws, min_periods=max(1, ws // 6)).sum()
        
        if new_features:
            X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
    except Exception:
        pass

    return X

def add_longrange_features(X, center_x, center_y, fps):
    """Long-range temporal features (windows & spans scaled by fps)."""
    new_features = {}
    for window in [120, 240]:
        ws = _scale(window, fps)
        if len(center_x) >= ws:
            new_features[f'x_ml{window}'] = center_x.rolling(ws, min_periods=max(5, ws // 6)).mean()
            new_features[f'y_ml{window}'] = center_y.rolling(ws, min_periods=max(5, ws // 6)).mean()

    # EWM spans also interpreted in frames
    for span in [60, 120]:
        s = _scale(span, fps)
        new_features[f'x_e{span}'] = center_x.ewm(span=s, min_periods=1).mean()
        new_features[f'y_e{span}'] = center_y.ewm(span=s, min_periods=1).mean()

    speed = safe_sqrt(center_x.diff()**2 + center_y.diff()**2) * float(fps)  # cm/s
    for window in [60, 120]:
        ws = _scale(window, fps)
        if len(speed) >= ws:
            new_features[f'sp_pct{window}'] = speed.rolling(ws, min_periods=max(5, ws // 6)).rank(pct=True)

    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_fft_features(X, signal, signal_name, fps, window_size=120):
    """
    Extract FFT-based frequency domain features using Vectorized Spectrogram.
    Massively faster than rolling loop.
    """
    ws = _scale(window_size, fps)
    N = len(signal)
    
    if N < ws:
        return X
    
    try:
        sig_filled = signal.ffill().bfill().fillna(0).values
        
        # Compute Spectrogram
        f, t, Sxx = spectrogram(sig_filled, fs=fps, window='hann', 
                                nperseg=ws, noverlap=ws-1, 
                                mode='psd', detrend='constant', scaling='density')
        
        # Sxx shape: (n_freqs, n_time_steps)
        # Transpose to (n_time_steps, n_freqs) to align with time
        Sxx = Sxx.T + 1e-20 
        
        # 1. Spectral Energy
        spectral_energy = np.sum(Sxx, axis=1)
        
        # 2. Spectral Entropy
        P_norm = Sxx / (spectral_energy[:, None])
        spectral_entropy = -np.sum(P_norm * np.log2(P_norm), axis=1)
        
        # 3. Dominant Frequency
        dom_idx = np.argmax(Sxx[:, 1:], axis=1) + 1
        dominant_freq = f[dom_idx]
        
        # 4.(0-1 Hz)
        mask_low = (f >= 0) & (f < 1.0)
        power_low = np.sum(Sxx[:, mask_low], axis=1)
        
        # 5.(1-5 Hz)
        mask_high = (f >= 1.0) & (f < 5.0)
        power_high = np.sum(Sxx[:, mask_high], axis=1)
        
        # Padding
        pad_head = ws // 2
        
        feats_dict = {
            f'{signal_name}_fft_domfreq': dominant_freq,
            f'{signal_name}_fft_energy': spectral_energy,
            f'{signal_name}_fft_entropy': spectral_entropy,
            f'{signal_name}_fft_power_low': power_low,
            f'{signal_name}_fft_power_high': power_high
        }
        
        new_features = {}
        valid_len = len(spectral_energy)
        
        for name, val_array in feats_dict.items():
            arr = np.full(N, np.nan)
            end_idx = min(N, pad_head + valid_len)
            arr[pad_head : end_idx] = val_array[:end_idx-pad_head]
            new_features[name] = arr
            
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)
        
    except Exception as e:
        print(f"FFT feature extraction failed: {e}")
    
    return X

def add_advanced_kinematics(X, center_x, center_y, fps):
    """Add acceleration, jerk, and angular velocity features."""
    # Velocity
    vel_x = center_x.diff() * fps  # cm/s
    vel_y = center_y.diff() * fps
    speed = safe_sqrt(vel_x**2 + vel_y**2)
    
    # Acceleration (cm/s^2)
    acc_x = vel_x.diff() * fps
    acc_y = vel_y.diff() * fps
    acc_mag = safe_sqrt(acc_x**2 + acc_y**2)
    
    # Jerk (cm/s^3)
    jerk_x = acc_x.diff() * fps
    jerk_y = acc_y.diff() * fps
    jerk_mag = safe_sqrt(jerk_x**2 + jerk_y**2)
    
    # Direction angle
    direction = np.arctan2(vel_y, vel_x)
    
    # Angular velocity (rad/s)
    angular_vel = direction.diff() * fps
    # Unwrap to handle -pi to pi discontinuities
    angular_vel = np.where(angular_vel > np.pi, angular_vel - 2*np.pi, angular_vel)
    angular_vel = np.where(angular_vel < -np.pi, angular_vel + 2*np.pi, angular_vel)
    
    # Add raw features
    new_features = {}
    new_features['speed'] = speed
    new_features['acc_mag'] = acc_mag
    new_features['jerk_mag'] = jerk_mag
    new_features['ang_vel'] = angular_vel
    
    # Add rolling statistics
    for w in [15, 30]:
        ws = _scale(w, fps)
        roll = dict(window=ws, min_periods=max(1, ws // 4))
        new_features[f'acc_mean_{w}'] = acc_mag.rolling(**roll).mean()
        new_features[f'acc_std_{w}'] = acc_mag.rolling(**roll).var().clip(lower=0).pow(0.5)
        new_features[f'jerk_mean_{w}'] = jerk_mag.rolling(**roll).mean()
        new_features[f'ang_vel_std_{w}'] = pd.Series(angular_vel).rolling(**roll).var().clip(lower=0).pow(0.5)
    
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_body_angle_features(X, nose_x, nose_y, body_x, body_y, tail_x, tail_y, fps):
    """Add head/body angle features relative to arena."""
    # Head direction (nose relative to body center)
    head_dir_x = nose_x - body_x
    head_dir_y = nose_y - body_y
    head_angle = np.arctan2(head_dir_y, head_dir_x)  # angle relative to arena
    
    # Body direction (body center relative to tail)
    body_dir_x = body_x - tail_x
    body_dir_y = body_y - tail_y
    body_angle = np.arctan2(body_dir_y, body_dir_x)
    
    # Angular velocities
    head_ang_vel = head_angle.diff() * fps
    body_ang_vel = body_angle.diff() * fps
    
    # Unwrap discontinuities
    head_ang_vel = np.where(head_ang_vel > np.pi, head_ang_vel - 2*np.pi, head_ang_vel)
    head_ang_vel = np.where(head_ang_vel < -np.pi, head_ang_vel + 2*np.pi, head_ang_vel)
    body_ang_vel = np.where(body_ang_vel > np.pi, body_ang_vel - 2*np.pi, body_ang_vel)
    body_ang_vel = np.where(body_ang_vel < -np.pi, body_ang_vel + 2*np.pi, body_ang_vel)
    
    new_features = {}
    new_features['head_angle'] = head_angle
    new_features['body_angle'] = body_angle
    new_features['head_ang_vel'] = head_ang_vel
    new_features['body_ang_vel'] = body_ang_vel
    
    # Rolling statistics
    for w in [15, 30]:
        ws = _scale(w, fps)
        new_features[f'head_ang_std_{w}'] = pd.Series(head_angle).rolling(ws, min_periods=max(1, ws//4)).var().clip(lower=0).pow(0.5)
        new_features[f'head_ang_vel_mean_{w}'] = pd.Series(head_ang_vel).rolling(ws, min_periods=max(1, ws//4)).mean()
    
    if new_features:
        X = pd.concat([X, pd.DataFrame(new_features, index=X.index)], axis=1)

    return X

def add_egocentric_features(X, mouse_df, fps):
    """
    Biến đổi tọa độ sang hệ quy chiếu lấy chuột làm tâm (Egocentric).
    Chuẩn hóa sao cho: Body Center tại (0,0), Mũi hướng về phía dương trục X.
    """
    # trục cơ thể
    if not all(p in mouse_df.columns.get_level_values(0) for p in ['nose', 'body_center']):
        return X

    # Vector từ tâm đến mũi
    dx = mouse_df['nose']['x'] - mouse_df['body_center']['x']
    dy = mouse_df['nose']['y'] - mouse_df['body_center']['y']
    
    # Góc quay của chuột so với trục hoành của camera
    angle = np.arctan2(dy, dx)
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)

    new_feats = {}
    
    # 2. Xoay tọa độ các bộ phận
    key_parts = ['ear_left', 'ear_right', 'tail_base', 'tail_tip']
    available_parts = mouse_df.columns.get_level_values(0)

    for part in key_parts:
        if part in available_parts:
            # Tọa độ tương đối so với tâm
            rx = mouse_df[part]['x'] - mouse_df['body_center']['x']
            ry = mouse_df[part]['y'] - mouse_df['body_center']['y']
            
            # x_new = x*cos - y*sin
            # y_new = x*sin + y*cos
            x_rot = rx * cos_a - ry * sin_a
            y_rot = rx * sin_a + ry * cos_a
            
            new_feats[f'ego_x_{part}'] = x_rot
            new_feats[f'ego_y_{part}'] = y_rot

    if new_feats:
        X = pd.concat([X, pd.DataFrame(new_feats, index=X.index)], axis=1)
        
    return X

def add_grooming_features(X, mouse_df, fps):
    """
    Phát hiện hành vi chải chuốt
    """
    if not all(p in mouse_df.columns.get_level_values(0) for p in ['nose', 'body_center']):
        return X

    # Tốc độ mũi
    nose_speed = np.sqrt(mouse_df['nose']['x'].diff()**2 + mouse_df['nose']['y'].diff()**2) * fps
    # Tốc độ thân
    body_speed = np.sqrt(mouse_df['body_center']['x'].diff()**2 + mouse_df['body_center']['y'].diff()**2) * fps

    # Tỷ lệ tách biệt
    decouple = nose_speed / (body_speed + 1e-3)
    
    w = int(0.5 * fps)
    
    new_feats = {}
    new_feats['head_body_ratio'] = decouple.rolling(w).median()
    new_feats['body_immobile'] = (body_speed < 2.0).astype(float) # < 2cm/s
    new_feats['nose_active'] = (nose_speed > 5.0).astype(float)   # > 5cm/s
    
    new_feats['grooming_score'] = new_feats['body_immobile'] * new_feats['nose_active']

    X = pd.concat([X, pd.DataFrame(new_feats, index=X.index)], axis=1)
    return X

def add_temporal_asymmetry(X, center_x, center_y, fps):
    """
    (Attack onset/offset).
    """
    speed = np.sqrt(center_x.diff()**2 + center_y.diff()**2) * fps
    
    # Cửa sổ 1 giây
    w = int(1.0 * fps)
    
    # Vận tốc trung bình Quá khứ
    # min_periods=1 để tránh NaN ở đầu video
    past_mean = speed.rolling(window=w, min_periods=1).mean()
    
    # Vận tốc trung bình Tương lai
    future_mean = speed.iloc[::-1].rolling(window=w, min_periods=1).mean().iloc[::-1]
    
    new_feats = {}
    # Delta V: Dương -> Đang tăng tốc
    new_feats['accel_trend_1s'] = future_mean - past_mean
    
    new_feats['accel_ratio_1s'] = future_mean / (past_mean + 1e-3)

    X = pd.concat([X, pd.DataFrame(new_feats, index=X.index)], axis=1)
    return X
