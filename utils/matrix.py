import numpy as np

def qvec_to_rotmat(qvec):
    """
    Convert a quaternion (w, x, y, z) to a 3x3 rotation matrix.

    Parameters:
    qvec (array-like): Quaternion in the form (w, x, y, z).

    Returns:
    np.ndarray: 3x3 rotation matrix.
    """
    w, x, y, z = qvec
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ])
    return R