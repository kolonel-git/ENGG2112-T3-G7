"""Compact deterministic report folder names for training experiments."""
import hashlib
import json


def training_report_name(config):
    """Keep exact experiment settings in the digest, rather than the path."""
    keys = ('alarm_codes', 'target_name', 'turbines', 'years', 'horizon', 'lookback',
            'train_end', 'validation_end', 'seed', 'selected_features', 'rolling_statistics')
    identity = {key: config[key] for key in keys if key in config}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:10]
    target = {'pitch_lubrication': 'pitch', 'untwisting_cables': 'untwisting'}.get(
        config['target_name'], config['target_name'])[:20]
    mode = 'selected' if 'selected_features' in config else 'all'
    # Duration strings are kept in the identity; minutes are supplied for display.
    return f"{target}_{config['horizon_minutes']}m_{mode}_{digest}"
