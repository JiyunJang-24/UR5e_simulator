"""Helpers for the OnRobot RG2 MuJoCo model (rg2.xml).

The actuator "fingers" (prefixed "rg2_fingers" when attached to the UR5e) is a position
servo on the outer-arm angle q [rad]: q = 0 is fully open, q = Q_CLOSE is closed.
The RG2 is a parallelogram linkage, so opening width is a sine of q, not linear.
"""
import json, os
import numpy as np

_P = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rg2_params.json")))
L, PHI_OPEN, GAP_OPEN, Q_CLOSE = _P["L"], _P["PHI_OPEN"], _P["GAP_OPEN"], _P["Q_CLOSE"]
PAD_Z_OPEN = _P["PAD_Z_OPEN"]          # pad centre distance from tool flange when open [m]
MAX_WIDTH = GAP_OPEN                   # ~0.106 m


def ctrl_to_width(q):
    """Finger-pad gap [m] for arm angle q [rad]."""
    return GAP_OPEN - 2 * L * (np.sin(PHI_OPEN + q) - np.sin(PHI_OPEN))


def width_to_ctrl(width):
    """Arm angle [rad] (actuator target) for a desired finger-pad gap [m]."""
    w = np.clip(width, 0.0, GAP_OPEN)
    return np.arcsin(np.sin(PHI_OPEN) + (GAP_OPEN - w) / (2 * L)) - PHI_OPEN


def pad_depth(q):
    """Distance of the pad centre from the tool flange along the tool axis [m].
    The fingertips move ~5 cm along the tool axis over the full stroke."""
    return PAD_Z_OPEN + L * (np.cos(PHI_OPEN + q) - np.cos(PHI_OPEN))


if __name__ == "__main__":
    for w in [0.106, 0.08, 0.05, 0.02, 0.0]:
        q = width_to_ctrl(w)
        print(f"width {w*1000:5.1f} mm -> ctrl {q:.4f} rad -> back {ctrl_to_width(q)*1000:5.1f} mm, pad depth {pad_depth(q)*1000:.1f} mm")
