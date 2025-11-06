import numpy as np


def steer(
    q_near: np.ndarray,
    q_rand: np.ndarray,
    step: float,
    bounds: np.ndarray,
) -> np.ndarray:
    """
    Move from q_near toward q_rand by at most `step`, clamped to `bounds`.

    Args:
        q_near: Current tree node position (shape: (D,)).
        q_rand: Target/sample position (shape: (D,)).
        step: Maximum step length toward q_rand.
        bounds: Workspace limits with shape (2, D): [[min...], [max...]].

    Returns:
        The proposed new position q_new (shape: (D,)).
    """
    direction = q_rand - q_near
    distance = float(np.linalg.norm(direction))

    if distance == 0.0:
        q_new = q_near
    elif distance <= step:
        q_new = q_rand
    else:
        q_new = q_near + (step / distance) * direction

    # Clamp to workspace bounds [[mins...], [maxes...]]
    q_new = np.clip(q_new, bounds[0], bounds[1])
    return q_new


