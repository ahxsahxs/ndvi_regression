"""Shared input adaptation for mapping dataset keys to model input names."""


def adapt_inputs(x, y):
    """Map dataset output dict keys to model input names.

    The dataset produces ``x['time']`` with shape ``(B, 10, 3)`` containing
    per-frame [year, sin_doy, cos_doy].  The model now expects the full
    sequence as ``temporal_metadata`` (shape ``(B, 10, 3)``).
    """
    return {
        'sentinel2_sequence': x['sentinel2'],
        'landcover_map': x['landcover'],
        'temporal_metadata': x['time'],
        'weather_sequence': x['weather'],
        'target_start_doy': x['target_start_doy'],
    }, y
