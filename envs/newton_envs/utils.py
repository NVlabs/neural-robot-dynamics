import warp as wp
import newton

@wp.func
def fmod(n: float, M: float):
    return ((n % M) + M) % M

@wp.func
def angle_normalize(x: float):
    return (fmod(x + wp.pi, 2.0 * wp.pi)) - wp.pi

@wp.kernel
def assign_controls(
    actions: wp.array(dtype=wp.float32, ndim=2),
    gains: wp.array(dtype=wp.float32),
    limits: wp.array(dtype=wp.float32, ndim=2),
    controllable_dofs: wp.array(dtype=wp.int32),
    control_full_dim: int,
    # outputs
    controls: wp.array(dtype=wp.float32),
):
    env_id = wp.tid()
    num_controls = gains.shape[0]
    for i in range(num_controls):
        lo = limits[i, 0]
        hi = limits[i, 1]
        idx = controllable_dofs[i]
        controls[env_id * control_full_dim + idx] = (
            wp.clamp(actions[env_id, i], lo, hi) * gains[i]
        )