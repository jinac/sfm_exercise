import jax
import jax.numpy as jnp
import jaxlie
import jaxls

@jax.jit
def project_point(
    point_world: jax.Array,
    T_camera_world: jaxlie.SE3,
    intrinsics: jax.Array,
) -> jax.Array:
    """Project 3D point to 2D using BAL camera model.

    Args:
        point_world: 3D point in world frame (3,)
        T_camera_world: Camera pose (world to camera transform)
        focal: Focal length

    Returns:
        2D projected point (2,)
    """
    
    # Transform point to camera frame.
    point_cam = T_camera_world @ point_world

    # Perspective projection (BAL convention: -P_xy / P_z)
    p = point_cam[:2] / point_cam[2]

    uv = intrinsics[:2] * p + intrinsics[2:]

    return uv

@jax.jit
def bal_params_to_se3(params: jax.Array) -> jaxlie.SE3:
    """Convert BAL camera parameters to SE3 pose.

    BAL stores rotation as Rodrigues vector (axis-angle).

    Args:
        params: Camera parameters [rodrigues(3), translation(3)]

    Returns:
        SE3 camera pose
    """
    return jaxlie.SE3.from_rotation_and_translation(
        rotation=jaxlie.SO3.exp(params[:3]),
        translation=params[3:6],
    )

class Point3Var(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(3)):
    """3D landmark position [x, y, z]."""

class CamIntrinsicsVar(jaxls.Var[jax.Array], default_factory=lambda: jnp.zeros(4)):
    """Cam Intrisics: [x, fy, cx, cy]"""

@jaxls.Cost.factory
def reprojection_cost(
    vals: jaxls.VarValues,
    camera_var: jaxls.SE3Var,
    intrinsics_var: CamIntrinsicsVar,
    point_var: Point3Var,
    observed_px: jax.Array,
    point_mask: jax.Array
) -> jax.Array:
    """Reprojection error with Huber loss for robustness."""
    intrinsics = vals[intrinsics_var]
    pose = vals[camera_var]
    point = vals[point_var]
    projected = project_point(point, pose, intrinsics)
    residual = projected - observed_px
    residual = point_mask * residual

    # IRLS-style Huber weighting for robustness to outliers.
    # For |r| <= delta: weight = 1 (quadratic region)
    # For |r| > delta: weight = delta / |r| (linear region)
    # stop_gradient prevents instabilities from differentiating through weights.
    delta = 2.0  # pixels
    abs_r = jnp.abs(residual) + 1e-8
    weight = jax.lax.stop_gradient(jnp.where(abs_r > delta, delta / abs_r, 1.0))
    return residual * jnp.sqrt(weight)

def ba(cams, cam_pose_tree, frames_meta, pts_3d, uv_coords, uv_masks):
    num_cams = len(cams)
    num_frames = len(frames_meta)
    num_points = pts_3d.shape[0]

    initial_poses = [
        bal_params_to_se3(cam_pose_tree.nodes[idx]["Rt"]) for idx in sorted(frames_meta.keys())
    ]
    # initial_poses = jax.vmap(bal_params_to_se3)(initial_poses)
    # pose_vars = jaxls.SE3Var(id=tuple(sorted(frames_meta.keys())))
    pose_vars = [jaxls.SE3Var(id=k) for k in sorted(frames_meta.keys())]
    initial_intrinsics = [jnp.array(cams[cam].params) for cam in sorted(cams.keys())]
    intrinsic_vars = [CamIntrinsicsVar(id=k) for k in sorted(cams.keys())]
    frame_to_idx = {k: idx for idx, k in enumerate(sorted(frames_meta.keys()))}

    # print(initial_poses)
    print(initial_intrinsics)

    # print(pose_vars)
    # print(intrinsic_vars)

    point_vars = Point3Var(id=jnp.arange(num_points))
    # point_vars = [Point3Var(id=i) for i in range(num_points)]
    # print(uv_coords[1])
    # print(uv_masks[1])
    # print(pts_3d)

    # uv_px, uv_mask = ba_data

    costs: list[jaxls.Cost] = [
        reprojection_cost(
            pose_vars[frame_to_idx[key]], # cam extrinsics
            intrinsic_vars[frames_meta[key].camera_id], # cam intrinsics
            point_vars, # 3d points
            jnp.array(uv_coords[key]),
            jnp.array(uv_masks[key])
        )
        for key in sorted(frames_meta.keys())
    ]
    # print(costs)

    initial_vals = jaxls.VarValues.make([
        # pose_vars.with_value(initial_poses),
        *[pose_vars[idx].with_value(pose) for idx, pose in enumerate(initial_poses)],
        *[intrinsic_vars[idx].with_value(intr) for idx, intr in enumerate(initial_intrinsics)],
        # *[point_vars[idx].with_value(pt) for idx, pt in enumerate(pts_3d)]
        point_vars.with_value(pts_3d)
    ])
    # print(initial_vals)
    # print(pose_vars[:5])
    # print(intrinsic_vars)
    # vars = [pose_vars + intrinsic_vars + [point_vars]]
    # print([point_vars])


    prob = jaxls.LeastSquaresProblem(costs, [*pose_vars, *intrinsic_vars, point_vars])
    prob = prob.analyze()

    solution = prob.solve(initial_vals, linear_solver="dense_cholesky")
    # print(solution)
    out_pts_3d = solution[Point3Var]

    return(out_pts_3d)