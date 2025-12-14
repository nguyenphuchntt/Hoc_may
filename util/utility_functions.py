import pandas as pd
import numpy as np
import warnings
from scipy.ndimage import gaussian_filter1d

def safe_sqrt(x):
    """Robust sqrt that clips negative values to 0."""
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return np.sqrt(x.clip(lower=0))
    return np.sqrt(np.maximum(x, 0))

# Suppress runtime warnings for invalid values in comparisons
warnings.filterwarnings('ignore', 'invalid value encountered')

def smooth_coordinates(df, sigma=1.5):
    """Apply Gaussian smoothing to coordinate columns."""
    df_smooth = df.copy()
    for col in df.columns:
        # Fill NaN before smoothing
        filled = df[col].interpolate(method='linear', limit_direction='both').fillna(0).values
        # Apply Gaussian filter
        smoothed = gaussian_filter1d(filled, sigma=sigma)
        df_smooth[col] = smoothed
    return df_smooth

def safe_rolling(series, window, func, min_periods=None):
    """Safe rolling operation with NaN handling"""
    if min_periods is None:
        min_periods = max(1, window // 4)
    return series.rolling(window, min_periods=min_periods, center=True).apply(func, raw=True)

def _scale(n_frames_at_30fps, fps, ref=30.0):
    """Scale a frame count defined at 30 fps to the current video's fps."""
    return max(1, int(round(n_frames_at_30fps * float(fps) / ref)))

def _scale_signed(n_frames_at_30fps, fps, ref=30.0):
    """Signed version of _scale for forward/backward shifts (keeps at least 1 frame when |n|>=1)."""
    if n_frames_at_30fps == 0:
        return 0
    s = 1 if n_frames_at_30fps > 0 else -1
    mag = max(1, int(round(abs(n_frames_at_30fps) * float(fps) / ref)))
    return s * mag

def _fps_from_meta(meta_df, fallback_lookup, default_fps=30.0):
    if 'frames_per_second' in meta_df.columns and pd.notnull(meta_df['frames_per_second']).any():
        return float(meta_df['frames_per_second'].iloc[0])
    vid = meta_df['video_id'].iloc[0]
    return float(fallback_lookup.get(vid, default_fps))

    