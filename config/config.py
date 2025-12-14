import os
import numpy as np
from xgboost import XGBClassifier

# GPU Configuration
def get_gpu_params():
    """Detect GPU and return XGBoost parameters."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True)
        if result.returncode == 0:
            return {'tree_method': 'hist', 'device': 'cuda', 'predictor': 'gpu_predictor'}
    except:
        pass
    return {'tree_method': 'hist', 'predictor': 'cpu_predictor'}

GPU_PARAMS = get_gpu_params()

# Main Configuration
class CFG:
    BASE_PATH = "/kaggle/input/MABe-mouse-behavior-detection"
    MODEL_SAVE_PATH = "./models"
    MODEL_LOAD_PATH = "./models"
    
    train_path = f"{BASE_PATH}/train.csv"
    test_path = f"{BASE_PATH}/test.csv"
    train_annotation_path = f"{BASE_PATH}/train_annotation"
    train_tracking_path = f"{BASE_PATH}/train_tracking"
    test_tracking_path = f"{BASE_PATH}/test_tracking"

    model_name = "xgboost"
    
    # XGBoost Model Config
    model = XGBClassifier(
        verbosity=0, 
        random_state=42,
        n_estimators=700, 
        learning_rate=0.05, 
        max_depth=6,
        min_child_weight=5, 
        subsample=0.8, 
        colsample_bytree=0.8,
        **GPU_PARAMS
    )

# CatBoost Configuration
CAT_PARAMS = {
    'iterations': 700,
    'learning_rate': 0.05,
    'depth': 7,
    'loss_function': 'Logloss',
    'verbose': 0,
    'random_seed': 42,
    'task_type': 'GPU' if GPU_PARAMS.get('device') == 'cuda' else 'CPU',
    'devices': '0',
    'allow_writing_files': False
}

# Data Processing Constants
drop_body_parts = [
    'headpiece_bottombackleft', 'headpiece_bottombackright', 'headpiece_bottomfrontleft', 'headpiece_bottomfrontright', 
    'headpiece_topbackleft', 'headpiece_topbackright', 'headpiece_topfrontleft', 'headpiece_topfrontright', 
    'spine_1', 'spine_2',
    'tail_middle_1', 'tail_middle_2', 'tail_midpoint'
]

# Optimized Thresholds
# Keys are JSON strings of body_parts_tracked.
_OPTIMIZED_THRESHOLDS = {
    '["body_center", "ear_left", "ear_right", "headpiece_bottombackleft", "headpiece_bottombackright", "headpiece_bottomfrontleft", "headpiece_bottomfrontright", "headpiece_topbackleft", "headpiece_topbackright", "headpiece_topfrontleft", "headpiece_topfrontright", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]': {
        'rear': 0.1802242836293328,
        'approach': 0.6539032604599244,
        'attack': 0.7847216373473394,
        'avoid': 0.6736294848811701,
        'chase': 0.725129014779558,
        'chaseattack': 0.7335692270948772,
        'submit': 0.7890928781700324
    },
    '["body_center", "ear_left", "ear_right", "hip_left", "hip_right", "lateral_left", "lateral_right", "nose", "spine_1", "spine_2", "tail_base", "tail_middle_1", "tail_middle_2", "tail_tip"]': {
        'huddle': 0.19324432235408093,
        'reciprocalsniff': 0.2846993153483479,
        'sniffgenital': 0.4216179622134541
    },
    '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "neck", "nose", "tail_base", "tail_midpoint", "tail_tip"]': {
        'rear': 0.354945509352908,
        'approach': 0.79409702749665,
        'attack': 0.42014539025183806,
        'avoid': 0.5155551356895803,
        'chase': 0.26775941251161717,
        'chaseattack': 0.6340882510525886,
        'submit': 0.3692337034773341
    },
    '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base", "tail_tip"]': {
        'attack': 0.6441354821664264,
        'dominance': 0.1838563642881675,
        'sniff': 0.12689889907052312,
        'chase': 0.6104653904205564,
        'escape': 0.7842697292560393,
        'follow': 0.5571638518400341
    },
    '["body_center", "ear_left", "ear_right", "lateral_left", "lateral_right", "nose", "tail_base"]': {
        'attack': 0.3273302612483364,
        'sniff': 0.38438391675198236,
        'defend': 0.46988959948717757,
        'escape': 0.4144199685146677,
        'mount': 0.3323015425388464
    },
    '["body_center", "ear_left", "ear_right", "nose", "tail_base"]': {
        'biteobject': 0.18632261293388278,
        'climb': 0.11928699503620606,
        'dig': 0.31221899091420324,
        'exploreobject': 0.21547934943872898,
        'rear': 0.20019244464149918,
        'selfgroom': 0.38625900388833434,
        'shepherd': 0.7004386318706047,
        'approach': 0.460343565212731,
        'attack': 0.5002182643492584,
        'chase': 0.628466891645852,
        'defend': 0.41070407036797685,
        'escape': 0.645515420225138,
        'flinch': 0.5309188128818834,
        'follow': 0.393561478566473,
        'sniff': 0.38515435819025323,
        'sniffface': 0.4853425328353815,
        'sniffgenital': 0.5150217141413699,
        'tussle': 0.19026751098389777
    },
    '["ear_left", "ear_right", "head", "tail_base"]': {
        'rear': 0.33247813184084696,
        'rest': 0.18212949085144509,
        'selfgroom': 0.3593606926950563,
        'climb': 0.2407105227943975,
        'dig': 0.37394356716099575,
        'run': 0.5876780903897704,
        'sniff': 0.29602832622578085,
        'sniffgenital': 0.4552036555988188,
        'approach': 0.3397955167565912,
        'defend': 0.5082239745304779,
        'escape': 0.5737342061656768,
        'attemptmount': 0.6784552674538017
    },
    '["ear_left", "ear_right", "hip_left", "hip_right", "neck", "nose", "tail_base"]': {
        'rear': 0.468610352931178,
        'selfgroom': 0.4271603226799843,
        'genitalgroom': 0.40163513830495,
        'dig': 0.34263893755148556,
        'approach': 0.4315636211201036,
        'attack': 0.32402249248559756,
        'disengage': 0.4211254889259497,
        'mount': 0.309481040813445,
        'sniff': 0.21935418775972404,
        'sniffgenital': 0.19240568098649966,
        'dominancemount': 0.5511767872192154,
        'sniffbody': 0.30430347433072624,
        'sniffface': 0.45050301260875103,
        'attemptmount': 0.32752351817104797,
        'intromit': 0.513169488895239,
        'chase': 0.10544678034988869,
        'escape': 0.6890486051086231,
        'reciprocalsniff': 0.30097880420880474,
        'allogroom': 0.39212894593546305,
        'ejaculate': 0.14998255706148433,
        'dominancegroom': 0.30233177236242303
    },
    '["ear_left", "ear_right", "nose", "tail_base", "tail_tip"]': {
        'freeze': 0.12347071422587626,
        'rear': 0.12725968189652528,
        'approach': 0.16420986895829462,
        'attack': 0.47103605544070304,
        'defend': 0.27920524419463066,
        'escape': 0.4917106399541633,
        'sniff': 0.17955329063463377
    },
}

DEFAULT_THRESHOLD = 0.27
